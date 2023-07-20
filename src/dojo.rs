use crate::car::Car;
use crate::car::Model;
use crate::car::SpawnCar;
use crate::car::UpdateCar;
use crate::configs;
use crate::enemy::SpawnEnemies;
use crate::enemy::UpdateEnemy;
use crate::ROAD_X_MIN;
use bevy::ecs::system::SystemState;
use bevy::log;
use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use bevy_tokio_tasks::TaskContext;
use bevy_tokio_tasks::{TokioTasksPlugin, TokioTasksRuntime};
use dojo_client::contract::world::WorldContract;
use num::bigint::BigUint;
use num::{FromPrimitive, ToPrimitive};
use rand::Rng;
use starknet::accounts::SingleOwnerAccount;
use starknet::core::types::{BlockId, BlockTag, FieldElement};
use starknet::core::utils::cairo_short_string_to_felt;
use starknet::providers::jsonrpc::HttpTransport;
use starknet::providers::JsonRpcClient;
use starknet::signers::{LocalWallet, SigningKey};
use std::ops::Div;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::mpsc;
use url::Url;

pub fn rand_felt_fixed_point() -> FieldElement {
    let mut rng = rand::thread_rng();
    ((rng.gen::<u128>() % 200) << 64).into()
}

#[derive(Resource)]
pub struct DojoEnv {
    /// The block ID to use for all contract calls.
    block_id: BlockId,
    /// The address of the world contract.
    world_address: FieldElement,
    /// The account to use for performing execution on the World contract.
    account: Arc<SingleOwnerAccount<JsonRpcClient<HttpTransport>, LocalWallet>>,
}

impl DojoEnv {
    fn new(
        world_address: FieldElement,
        account: SingleOwnerAccount<JsonRpcClient<HttpTransport>, LocalWallet>,
    ) -> Self {
        Self {
            world_address,
            account: Arc::new(account),
            block_id: BlockId::Tag(BlockTag::Latest),
        }
    }
}

pub struct DojoPlugin;

impl Plugin for DojoPlugin {
    fn build(&self, app: &mut App) {
        let url = Url::parse(configs::JSON_RPC_ENDPOINT).unwrap();
        let account_address = FieldElement::from_str(configs::ACCOUNT_ADDRESS).unwrap();
        let account = SingleOwnerAccount::new(
            JsonRpcClient::new(HttpTransport::new(url)),
            LocalWallet::from_signing_key(SigningKey::from_secret_scalar(
                FieldElement::from_str(configs::ACCOUNT_SECRET_KEY).unwrap(),
            )),
            account_address,
            cairo_short_string_to_felt("KATANA").unwrap(),
        );

        let world_address = FieldElement::from_str(configs::WORLD_ADDRESS).unwrap();

        app.add_plugin(TokioTasksPlugin::default())
            .insert_resource(DojoEnv::new(world_address, account))
            .add_startup_systems((
                setup,
                spawn_racers_thread,
                drive_thread,
                update_vehicle_thread,
                update_enemies_thread,
            ))
            .add_system(sync_dojo_state);
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(DojoSyncTime::from_seconds(configs::DOJO_SYNC_INTERVAL));
}

#[derive(Component)]
struct DojoSyncTime {
    timer: Timer,
}

impl DojoSyncTime {
    fn from_seconds(duration: f32) -> Self {
        Self {
            timer: Timer::from_seconds(duration, TimerMode::Repeating),
        }
    }
}

fn sync_dojo_state(
    mut dojo_sync_time: Query<&mut DojoSyncTime>,
    time: Res<Time>,
    drive: Res<DriveCommand>,
    update_vehicle: Res<UpdateVehicleCommand>,
    update_enemies: Res<UpdateEnemiesCommand>,
    spawn_racers: Res<SpawnRacersCommand>,
    cars: Query<&Collider, With<Car>>,
) {
    let mut dojo_time = dojo_sync_time.single_mut();

    if dojo_time.timer.just_finished() {
        dojo_time.timer.reset();
        if cars.is_empty() {
            if let Err(e) = spawn_racers.try_send() {
                log::error!("Spawn racers channel: {e}");
            }
        } else {
            if let Err(e) = update_vehicle.try_send() {
                log::error!("Update vehicle channel: {e}");
            }
            if let Err(e) = drive.try_send() {
                log::error!("Drive channel: {e}");
            }
            if let Err(e) = update_enemies.try_send() {
                log::error!("Update enemies channel: {e}");
            }
        }
    } else {
        dojo_time.timer.tick(time.delta());
    }
}

fn spawn_racers_thread(
    env: Res<DojoEnv>,
    runtime: ResMut<TokioTasksRuntime>,
    mut commands: Commands,
) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(SpawnRacersCommand(tx));

    let account = env.account.clone();
    let world_address = env.world_address;
    let block_id = env.block_id;

    runtime.spawn_background_task(move |mut ctx| async move {
        let world = WorldContract::new(world_address, account.as_ref());
        let spawn_racer_system = world.system("spawn_racer", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let model_id = cairo_short_string_to_felt(configs::MODEL_NAME).unwrap();

            match spawn_racer_system
                .execute(vec![
                    model_id,
                    rand_felt_fixed_point(),
                    FieldElement::ZERO,
                    FieldElement::ZERO,
                    FieldElement::ZERO,
                ])
                .await
            {
                Ok(_) => {
                    ctx.run_on_main_thread(move |ctx| {
                        let mut state: SystemState<(
                            EventWriter<SpawnCar>,
                            EventWriter<SpawnEnemies>,
                        )> = SystemState::new(ctx.world);
                        let (mut spawn_car, mut spawn_enemies) = state.get_mut(ctx.world);

                        spawn_enemies.send(SpawnEnemies);
                        spawn_car.send(SpawnCar);
                    })
                    .await;
                }
                Err(e) => {
                    log::error!("Run spawn_racer system: {e}");
                }
            }
        }
    });
}

fn drive_thread(env: Res<DojoEnv>, runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(DriveCommand(tx));

    let account = env.account.clone();
    let world_address = env.world_address;
    let block_id = env.block_id;

    runtime.spawn_background_task(move |ctx| async move {
        let world = WorldContract::new(world_address, account.as_ref());

        let drive_system = world.system("drive", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let model_id = get_model_id(ctx.clone()).await;

            match model_id {
                Some(model_id) => {
                    if let Err(e) = drive_system.execute(vec![model_id]).await {
                        log::error!("Run drive system: {e}");
                    }
                }
                None => {}
            }
        }
    });
}

fn update_vehicle_thread(
    env: Res<DojoEnv>,
    runtime: ResMut<TokioTasksRuntime>,
    mut commands: Commands,
) {
    let (tx, mut rx) = mpsc::channel::<()>(16);
    commands.insert_resource(UpdateVehicleCommand(tx));

    let account = env.account.clone();
    let world_address = env.world_address;
    let block_id = env.block_id;

    runtime.spawn_background_task(move |mut ctx| async move {
        let world = WorldContract::new(world_address, account.as_ref());
        let vehicle_component = world.component("Vehicle", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let model_id = get_model_id(ctx.clone()).await;

            if let Some(model_id) = model_id {
                match vehicle_component
                    .entity(FieldElement::ZERO, vec![model_id], block_id)
                    .await
                {
                    Ok(vehicle) => {
                        ctx.run_on_main_thread(move |ctx| {
                            let mut state: SystemState<EventWriter<UpdateCar>> =
                                SystemState::new(ctx.world);
                            let mut update_car = state.get_mut(ctx.world);

                            update_car.send(UpdateCar { vehicle })
                        })
                        .await;
                    }
                    Err(e) => {
                        log::error!("Query `Vehicle` component: {e}");
                    }
                }
            }
        }
    });
}

fn update_enemies_thread(
    env: Res<DojoEnv>,
    runtime: ResMut<TokioTasksRuntime>,
    mut commands: Commands,
) {
    let (tx, mut rx) = mpsc::channel::<()>(16);
    commands.insert_resource(UpdateEnemiesCommand(tx));

    let account = env.account.clone();
    let world_address = env.world_address;
    let block_id = env.block_id;

    runtime.spawn_background_task(move |mut ctx| async move {
        let world = WorldContract::new(world_address, account.as_ref());
        let position_component = world.component("Position", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let model_id = get_model_id(ctx.clone()).await;

            if let Some(model_id) = model_id {
                // TODO: query multiple enemies at once
                for i in 0..configs::DOJO_ENEMIES_NB {
                    let enemy_id: FieldElement = i.into();

                    match position_component
                        .entity(
                            FieldElement::ZERO,
                            vec![model_id, enemy_id.into()],
                            block_id,
                        )
                        .await
                    {
                        Ok(position) => {
                            ctx.run_on_main_thread(move |ctx| {
                                let mut state: SystemState<EventWriter<UpdateEnemy>> =
                                    SystemState::new(ctx.world);
                                let mut update_enemy = state.get_mut(ctx.world);

                                update_enemy.send(UpdateEnemy { position, enemy_id })
                            })
                            .await
                        }
                        Err(e) => {
                            log::error!("Query `Position` component: {e}");
                        }
                    }
                }
            }
        }
    });
}

#[derive(Resource)]
pub struct SpawnRacersCommand(mpsc::Sender<()>);

// TODO: derive macro?
impl SpawnRacersCommand {
    pub fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
struct DriveCommand(mpsc::Sender<()>);

// TODO: derive macro?
impl DriveCommand {
    fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
struct UpdateVehicleCommand(mpsc::Sender<()>);

impl UpdateVehicleCommand {
    fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
pub struct UpdateEnemiesCommand(mpsc::Sender<()>);

impl UpdateEnemiesCommand {
    pub fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

pub fn fixed_to_f32(val: FieldElement) -> f32 {
    BigUint::from_str(&val.to_string())
        .unwrap()
        .div(BigUint::from_i8(2).unwrap().pow(64))
        .to_f32()
        .unwrap()
}

pub fn dojo_to_bevy_coordinate(dojo_x: f32, dojo_y: f32) -> (f32, f32) {
    let bevy_x = dojo_x * configs::DOJO_TO_BEVY_RATIO_X + ROAD_X_MIN;
    let bevy_y = dojo_y * configs::DOJO_TO_BEVY_RATIO_Y;

    // log::info!("dojo_x: {}, dojo_y: {}", dojo_x, dojo_y);
    // log::info!("bevy_x: {}, bevy_y: {}", bevy_x, bevy_y);

    (bevy_x, bevy_y)
}

async fn get_model_id(mut ctx: TaskContext) -> Option<FieldElement> {
    ctx.run_on_main_thread(move |ctx| {
        let mut state: SystemState<Query<&Model, With<Car>>> = SystemState::new(ctx.world);
        let query = state.get(ctx.world);

        match query.get_single() {
            Ok(model) => Some(model.id),
            Err(_) => None,
        }
    })
    .await
}
