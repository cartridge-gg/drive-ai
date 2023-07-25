use debug::PrintTrait;
use traits::Into;
use cubit::types::vec2::{Vec2, Vec2Trait};
use cubit::types::fixed::{Fixed, FixedTrait, ONE_u128};
use cubit::math::{trig, comp::{min, max}, core::{pow_int, sqrt}};
use starknet::ContractAddress;
use drive_ai::{Vehicle, VehicleTrait};
use drive_ai::enemy::{Position, PositionTrait};
use drive_ai::math::{intersects};
use drive_ai::rays::{RaysTrait, Rays, Ray, RayTrait, NUM_RAYS, RAY_LENGTH};
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core as orion_fp;
use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

#[derive(Component, Serde, SerdeLen, Drop, Copy)]
struct Racer {
    // Vehicle owner
    driver: ContractAddress,
    // Model system name
    model: felt252,
}

#[derive(Serde, Drop)]
struct Sensors {
    rays: Tensor<orion_fp::FixedType>, 
}

#[derive(Serde, Drop, PartialEq)]
enum Wall {
    None: (),
    Left: (),
    Right: (),
}

const GRID_HEIGHT: u128 = 18446744073709551616000; // 1000
const GRID_WIDTH: u128 = 7378697629483820646400; // 400
const HALF_GRID_WIDTH: u128 = 3689348814741910323200; // 200
const CAR_HEIGHT: u128 = 590295810358705651712; // 32
const CAR_WIDTH: u128 = 295147905179352825856; // 16

fn compute_sensors(vehicle: Vehicle, mut enemies: Array<Position>) -> Sensors {
    // All sensors (ray segments) for this vehicle
    let ray_segments = RaysTrait::new(vehicle.position, vehicle.steer).segments;

    // Distances along each sensor ray to near wall
    // If ray does not intersect wall, sensor's distance is 0
    // If no near wall, array is empty
    let mut wall_sensors = match near_wall(vehicle) {
        Wall::None(()) => {
            ArrayTrait::<Fixed>::new()
        },
        Wall::Left(()) => {
            distances_to_wall(Wall::Left(()), ray_segments)
        },
        Wall::Right(()) => {
            distances_to_wall(Wall::Right(()), ray_segments)
        },
    };

    // Positions of only filtered (near) enemies
    let filtered_enemies = filter_positions(vehicle, enemies.span());

    // Distances along each sensor ray to nearest intersection of ray w/edge of filtered enemy
    // If sensor does not intersect an enemy edge, sensor's distance = 0
    let mut enemy_sensors = ArrayTrait::<Fixed>::new();
    let mut ray_idx = 0;
    loop {
        if (ray_idx == NUM_RAYS) {
            break ();
        }

        enemy_sensors.append(closest_position(ray_segments.at(ray_idx), filtered_enemies.span()));

        ray_idx += 1;
    };

    let mut sensors = ArrayTrait::<orion_fp::FixedType>::new();

    let mut idx = 0;
    if wall_sensors.len() > 0 {
        loop {
            if idx == NUM_RAYS {
                break ();
            }

            let wall_sensor = *wall_sensors.at(idx);
            let enemy_sensor = *enemy_sensors.at(idx);

            if wall_sensor < enemy_sensor {
                sensors.append(orion_fp::FixedTrait::new(wall_sensor.mag, false));
            } else {
                sensors.append(orion_fp::FixedTrait::new(enemy_sensor.mag, false));
            };

            idx += 1;
        }
    } else {
        loop {
            if idx == NUM_RAYS {
                break ();
            }

            sensors.append(orion_fp::FixedTrait::new(*enemy_sensors.at(idx).mag, false));

            idx += 1;
        }
    }

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    let extra = Option::<ExtraParams>::None(());
    Sensors { rays: TensorTrait::new(shape.span(), sensors.span(), extra) }
}

fn filter_positions(vehicle: Vehicle, mut positions: Span<Position>) -> Array<Position> {
    // Will hold near position values
    let mut near = ArrayTrait::new();

    // Filter is rectangle around vehicle
    // Includes only enemies w/position (center) inside rectangle
    // Since enemy has zero rotation, max vehicle-to-enemy center-to-center distances:
    // Max horizontal distance = RAY_LENGTH + enemy's CAR_WIDTH
    let filter_dist_x = FixedTrait::new(RAY_LENGTH + CAR_WIDTH, false);
    // Max vertical distance = RAY_LENGTH + enemy's CAR_HEIGHT
    let filter_dist_y = FixedTrait::new(RAY_LENGTH + CAR_HEIGHT, false);

    loop {
        match positions.pop_front() {
            Option::Some(position) => {
                if (FixedTrait::new(*position.x, false) - vehicle.position.x).abs() <= filter_dist_x
                    && (FixedTrait::new(*position.y, false) - vehicle.position.y)
                        .abs() <= filter_dist_y {
                    near.append(*position);
                }
            },
            Option::None(_) => {
                break ();
            }
        };
    };

    near
}

// Returns distances of sensor (ray) to closest intersecting edge of all filtered enemies
// (If sensor does not intersect an enemy edge, sensor's distance = 0)
fn closest_position(ray: @Ray, mut positions: Span<Position>) -> Fixed {
    // Set to max possible intersection distance before outer loop
    let mut closest = FixedTrait::new(RAY_LENGTH, false);
    loop {
        match positions.pop_front() {
            Option::Some(position) => {
                let mut edge_idx: usize = 0;

                let vertices = position.vertices_scaled();

                // TODO: Only check visible edges
                loop {
                    if edge_idx == 4 {
                        break ();
                    }

                    // Endpoints of edge
                    let p2 = vertices.at(edge_idx);
                    let mut q2_idx = edge_idx + 1;
                    if q2_idx == 4 {
                        q2_idx = 0;
                    }

                    let q2 = vertices.at(q2_idx);
                    if ray.intersects(*p2, *q2) {
                        let dist = ray.dist(*p2, *q2);
                        if dist < closest {
                            closest = dist;
                        }
                    }

                    edge_idx += 1;
                }
            },
            Option::None(_) => {
                if closest == FixedTrait::new(RAY_LENGTH, false) {
                    // No intersection found for ray
                    closest = FixedTrait::new(0, false);
                }
                break ();
            }
        };
    };

    closest
}

fn near_wall(vehicle: Vehicle) -> Wall {
    if vehicle.position.x <= FixedTrait::new(RAY_LENGTH, false) {
        return Wall::Left(());
    } else if vehicle.position.x >= FixedTrait::new(GRID_WIDTH - RAY_LENGTH, false) {
        return Wall::Right(());
    }
    return Wall::None(());
}

// Distances along each sensor ray to near wall
// If ray does not intersect wall, sensor's distance is 0
// If no near wall, array is empty
fn distances_to_wall(near_wall: Wall, mut rays: Span<Ray>) -> Array<Fixed> {
    let mut sensors = ArrayTrait::<Fixed>::new();

    match near_wall {
        Wall::None(()) => {
            return sensors;
        },
        Wall::Left(()) => {
            // Wall endpoints
            let p2 = Vec2 { x: FixedTrait::new(0, false), y: FixedTrait::new(0, false) };
            let q2 = Vec2 { x: FixedTrait::new(0, false), y: FixedTrait::new(GRID_HEIGHT, false) };
            loop {
                match rays.pop_front() {
                    Option::Some(ray) => {
                        // If ray.q.x is negative or zero, ray intersects with Left wall
                        //     Cheaper than ray.intersects(p2, q2)?
                        if *(ray.q.x.sign) || *(ray.q.x.mag) == 0_u128 {
                            sensors.append(ray.dist(p2, q2));
                        } else {
                            sensors.append(FixedTrait::new(0, false));
                        }
                    },
                    Option::None(_) => {
                        break ();
                    }
                };
            };
        },
        Wall::Right(()) => {
            // Wall endpoints
            let p2 = Vec2 { x: FixedTrait::new(GRID_WIDTH, false), y: FixedTrait::new(0, false) };
            let q2 = Vec2 {
                x: FixedTrait::new(GRID_WIDTH, false), y: FixedTrait::new(GRID_HEIGHT, false)
            };
            loop {
                match rays.pop_front() {
                    Option::Some(ray) => {
                        // If ray.q.x is >= GRID_WIDTH, ray intersects with Right wall
                        if !*(ray.q.x.sign) && *(ray.q.x.mag) >= GRID_WIDTH {
                            sensors.append(ray.dist(p2, q2));
                        } else {
                            sensors.append(FixedTrait::new(0, false));
                        }
                    },
                    Option::None(_) => {
                        break ();
                    }
                };
            };
        },
    };
    sensors
}

fn collision_check(vehicle: Vehicle, mut enemies: Array<Position>) {
    let vertices = vehicle.vertices();

    /// Wall collision check
    match near_wall(vehicle) {
        Wall::None(()) => {},
        Wall::Left(()) => { // not 100% sure of syntax here at end
            let cos_theta = trig::cos_fast(vehicle.steer);
            let sin_theta = trig::sin_fast(vehicle.steer);

            // Check only left edge (vertex 1 to 2)
            let closest_edge = Ray {
                theta: vehicle.steer,
                cos_theta: cos_theta,
                sin_theta: sin_theta,
                p: *vertices.at(1),
                q: *vertices.at(2),
            };
            let p2 = Vec2 { x: FixedTrait::new(0, false), y: FixedTrait::new(0, false) };
            let q2 = Vec2 { x: FixedTrait::new(0, false), y: FixedTrait::new(GRID_HEIGHT, false) };

            assert(!closest_edge.intersects(p2, q2), 'hit left wall');
        },
        Wall::Right(()) => { // not 100% sure of syntax here at end
            let cos_theta = trig::cos_fast(vehicle.steer);
            let sin_theta = trig::sin_fast(vehicle.steer);

            // Check only right edge (vertex 3 to 0)
            let closest_edge = Ray {
                theta: vehicle.steer,
                cos_theta: cos_theta,
                sin_theta: sin_theta,
                p: *vertices.at(3),
                q: *vertices.at(0),
            };

            let p2 = Vec2 { x: FixedTrait::new(GRID_WIDTH, false), y: FixedTrait::new(0, false) };
            let q2 = Vec2 {
                x: FixedTrait::new(GRID_WIDTH, false), y: FixedTrait::new(GRID_HEIGHT, false)
            };

            assert(!closest_edge.intersects(p2, q2), 'hit right wall');
        },
    };

    /// Enemy collision check
    // Get array of only near enemies positions
    let mut filtered_enemies = filter_positions(vehicle, enemies.span());

    // For each vehicle edge...
    let mut vehicle_edge_idx: usize = 0;
    loop {
        if (vehicle_edge_idx == 3) {
            break ();
        }

        let mut q1_idx = vehicle_edge_idx + 1;
        if q1_idx == 4 {
            q1_idx = 0;
        }
        // Endpoints of vehicle edge
        let p1 = vertices.at(vehicle_edge_idx);
        let q1 = vertices.at(q1_idx);

        // ..., check for collision with each near enemy
        loop {
            match filtered_enemies.pop_front() {
                Option::Some(position) => {
                    let mut enemy_edge_idx: usize = 0;

                    let vertices = position.vertices_scaled();

                    // For each enemy edge
                    // TODO: Only check visible edges
                    loop {
                        if enemy_edge_idx == 3 {
                            break ();
                        }

                        let mut q2_idx = enemy_edge_idx + 1;
                        if q2_idx == 4 {
                            q2_idx = 0;
                        }

                        // Endpoints of enemy edge
                        let p2 = vertices.at(enemy_edge_idx);
                        let q2 = vertices.at(q2_idx);

                        assert(!intersects(*p1, *q1, *p2, *q2), 'hit enemy');

                        enemy_edge_idx += 1;
                    }
                },
                Option::None(_) => {
                    break ();
                }
            };
        };
        vehicle_edge_idx += 1;
    };
}


#[system]
mod spawn_racer {
    use array::ArrayTrait;
    use traits::Into;
    use cubit::types::FixedTrait;
    use cubit::types::{Vec2Trait, Vec2};

    use dojo::world::Context;
    use drive_ai::Vehicle;

    use super::{Racer, HALF_GRID_WIDTH};

    const FIFTY: u128 = 922337203685477580800;

    fn execute(ctx: Context, model: felt252, position: Vec2) {
        // let position = Vec2Trait::new(
        //     FixedTrait::new(HALF_GRID_WIDTH, false), FixedTrait::new(0, false)
        // );
        set !(
            ctx.world,
            model.into(),
            (
                Racer {
                    driver: ctx.origin, model
                    }, Vehicle {
                    position,
                    steer: FixedTrait::new(0_u128, false),
                    speed: FixedTrait::new(FIFTY, false),
                }
            )
        );

        let mut calldata = ArrayTrait::new();
        calldata.append(model);
        ctx.world.execute('spawn_enemies', calldata.span());

        return ();
    }
}

#[system]
mod drive {
    use array::ArrayTrait;
    use traits::Into;
    use serde::Serde;
    use dojo::world::Context;
    use drive_ai::vehicle::{Controls, Vehicle, VehicleTrait};
    use drive_ai::enemy::{Position, ENEMIES_NB};
    use super::{Racer, Sensors, compute_sensors};

    fn execute(ctx: Context, model: felt252) {
        let mut vehicle = get !(ctx.world, model.into(), Vehicle);

        let mut enemies = ArrayTrait::<Position>::new();
        let mut i: u8 = 0;
        loop {
            if i == ENEMIES_NB {
                break ();
            }
            let key = (model, i).into();
            let position = get !(ctx.world, key, Position);
            enemies.append(position);
            i += 1;
        }

        // 1. Compute sensors, reverts if there is a collision (game over)
        let sensors = compute_sensors(vehicle, enemies);
        // 2. Run model forward pass
        let mut sensor_calldata = ArrayTrait::new();
        sensors.serialize(ref sensor_calldata);
        let mut controls = ctx.world.execute('model', sensor_calldata.span());
        let controls = serde::Serde::<Controls>::deserialize(ref controls).unwrap();
        // 3. Update car position
        vehicle.control(controls);
        vehicle.drive();
        set !(
            ctx.world,
            model.into(),
            (Vehicle { position: vehicle.position, steer: vehicle.steer, speed: vehicle.speed })
        );

        // 4. Move enemeies to updated positions
        // TODO: This retrieves enemies again internally, we should
        // only read them once (pass them in here?)
        let mut calldata = ArrayTrait::new();
        calldata.append(model);
        ctx.world.execute('move_enemies', calldata.span());
    }
}

#[cfg(test)]
mod tests {
    use debug::PrintTrait;
    use array::{ArrayTrait, SpanTrait};
    use cubit::types::vec2::{Vec2, Vec2Trait};
    use cubit::types::fixed::{Fixed, FixedTrait, FixedPrint, ONE_u128};
    use cubit::math::trig;
    use cubit::test::helpers::assert_precise;
    use drive_ai::vehicle::{Vehicle, VehicleTrait};
    use drive_ai::rays::{Rays, RaysTrait, Ray, RayTrait, RAY_LENGTH};
    use drive_ai::enemy::{Position, PositionTrait};
    use drive_ai::math::assert_precise_u128;
    use super::{
        compute_sensors, filter_positions, closest_position, near_wall, distances_to_wall,
        collision_check, Wall
    };
    use super::{GRID_HEIGHT, GRID_WIDTH, CAR_HEIGHT, CAR_WIDTH};

    const NUM_RAYS: usize = 5; // Asserted values below are only for NUM_RAYS = 5
    const TWO: u128 = 36893488147419103232;
    const TEN: u128 = 184467440737095516160;
    const FIFTY: u128 = 922337203685477580800;
    const HUNDRED: u128 = 1844674407370955161600;
    const TWO_HUNDRED: u128 = 3689348814741910323200;
    const THREE_HUNDRED: u128 = 5534023222112865484800;
    const THREE_FIFTY: u128 = 6456360425798343065600;
    const DEG_25_IN_RADS: u128 = 8048910508974580935;

    #[test]
    #[available_gas(20000000)]
    fn test_compute_sensors() {}

    #[test]
    #[available_gas(20000000)]
    fn test_filter_positions() {
        // Spawn enemies, spaced evenly horizontally, a little ahead of vehicles at y = TWO_HUNDRED
        // Use scaled u128 for Position
        let enemies_nb = 4_u128;
        // position.x of first enemy (center), so left edge is `CAR_WIDTH` (half-width) from left wall
        let enemy_min_dist_from_wall = 2 * CAR_WIDTH;
        let enemy_horiz_spacing = (GRID_WIDTH - 2 * enemy_min_dist_from_wall) / (enemies_nb - 1);

        let mut enemies = ArrayTrait::<Position>::new();
        let mut i = 0_u128;
        loop {
            if i == enemies_nb {
                break ();
            }
            let x = enemy_min_dist_from_wall + i * enemy_horiz_spacing;
            let y = THREE_HUNDRED;
            enemies.append(Position { x: x, y: y });
            i += 1;
        };

        // Vehicle 1
        let vehicle_1 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(HUNDRED, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };

        let filtered_enemies_1 = filter_positions(vehicle_1, enemies.span());

        // Values calculated in spreadsheet "drive_ai tests"
        assert(filtered_enemies_1.len() == 3_usize, 'invalid filtered_enemies_1');
        assert_precise_u128(
            *(filtered_enemies_1.at(0).x),
            590295810358704000000,
            'invalid filtered_enemies_1 0 x',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_1.at(0).y),
            5534023222112850000000,
            'invalid filtered_enemies_1 0 y',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_1.at(1).x),
            2656331146614170000000,
            'invalid filtered_enemies_1 1 x',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_1.at(1).y),
            5534023222112850000000,
            'invalid filtered_enemies_1 1 y',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_1.at(2).x),
            4722366482869630000000,
            'invalid filtered_enemies_1 2 x',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_1.at(2).y),
            5534023222112850000000,
            'invalid filtered_enemies_1 2 y',
            Option::None(())
        );

        // Vehicle 2
        let vehicle_2 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(THREE_FIFTY, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(DEG_25_IN_RADS, false),
            speed: FixedTrait::new(0, false)
        };

        let filtered_enemies_2 = filter_positions(vehicle_2, enemies.span());
        assert(filtered_enemies_2.len() == 2_usize, 'invalid filtered_enemies_2');
        assert_precise_u128(
            *(filtered_enemies_2.at(0).x),
            4722366482869630000000,
            'invalid filtered_enemies_2 0 x',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_2.at(0).y),
            5534023222112850000000,
            'invalid filtered_enemies_2 0 y',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_2.at(1).x),
            6788401819125100000000,
            'invalid filtered_enemies_2 1 x',
            Option::None(())
        );
        assert_precise_u128(
            *(filtered_enemies_2.at(1).y),
            5534023222112850000000,
            'invalid filtered_enemies_2 1 y',
            Option::None(())
        );
    }

    #[test]
    #[available_gas(200000000)] // Made gas 10x 
    fn test_closest_position() {
        // Spawn enemies, spaced evenly horizontally, a little ahead of vehicles at y = TWO_HUNDRED
        // Use scaled u128 for Position
        let enemies_nb = 4_u128;
        // position.x of first enemy (center), so left edge is `CAR_WIDTH` (half-width) from left wall
        let enemy_min_dist_from_wall = 2 * CAR_WIDTH;
        let enemy_horiz_spacing = (GRID_WIDTH - 2 * enemy_min_dist_from_wall) / (enemies_nb - 1);

        let mut enemies = ArrayTrait::<Position>::new();
        let mut i = 0_u128;
        loop {
            if i == enemies_nb {
                break ();
            }
            let x = enemy_min_dist_from_wall + i * enemy_horiz_spacing;
            let y = THREE_HUNDRED;
            enemies.append(Position { x: x, y: y });
            i += 1;
        };

        // Vehicle 1
        let vehicle_1 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(HUNDRED, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };

        let ray_segments_1 = RaysTrait::new(vehicle_1.position, vehicle_1.steer).segments;

        let filtered_enemies_1 = filter_positions(vehicle_1, enemies.span());

        let mut enemy_sensors_1 = ArrayTrait::<Fixed>::new();
        let mut ray_idx = 0;
        loop {
            if (ray_idx == NUM_RAYS) {
                break ();
            }

            enemy_sensors_1
                .append(closest_position(ray_segments_1.at(ray_idx), filtered_enemies_1.span()));

            ray_idx += 1;
        };

        // Values calculated in spreadsheet "drive_ai tests"
        // Asserted values below are only for NUM_RAYS = 5
        assert(enemy_sensors_1.len() == NUM_RAYS, 'invalid enemy_sensors_1');
        assert_precise(
            *(enemy_sensors_1.at(0)),
            1951466671275690000000,
            'invalid v1 closest pos ray 0',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(enemy_sensors_1.at(1)),
            1918461383665790000000,
            'invalid v1 closest pos ray 1',
            Option::Some(184467440737095000) // 1e-02
        );
        assert(*(enemy_sensors_1.at(2).mag) == 0, 'invalid v1 closest pos ray 2');
        assert_precise(
            *(enemy_sensors_1.at(3)),
            1448431641301450000000,
            'invalid v1 closest pos ray 3',
            Option::Some(184467440737095000) // 1e-02
        );
        assert(*(enemy_sensors_1.at(4).mag) == 0, 'invalid v1 closest pos ray 4');

        // Vehicle 2
        let vehicle_2 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(THREE_FIFTY, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(DEG_25_IN_RADS, false),
            speed: FixedTrait::new(0, false)
        };

        let ray_segments_2 = RaysTrait::new(vehicle_2.position, vehicle_2.steer).segments;

        let filtered_enemies_2 = filter_positions(vehicle_2, enemies.span());

        let mut enemy_sensors_2 = ArrayTrait::<Fixed>::new();
        ray_idx = 0;
        loop {
            if (ray_idx == NUM_RAYS) {
                break ();
            }

            enemy_sensors_2
                .append(closest_position(ray_segments_2.at(ray_idx), filtered_enemies_2.span()));

            ray_idx += 1;
        };
        assert(enemy_sensors_2.len() == NUM_RAYS, 'invalid enemy_sensors_2');
        assert(*(enemy_sensors_2.at(0).mag) == 0, 'invalid v2 closest pos ray 0');
        assert(*(enemy_sensors_2.at(1).mag) == 0, 'invalid v2 closest pos ray 1');
        assert_precise(
            *(enemy_sensors_2.at(2)),
            1384053645962460000000,
            'invalid v2 closest pos ray 2',
            Option::Some(184467440737095000) // 1e-02
        );
        assert(*(enemy_sensors_2.at(3).mag) == 0, 'invalid v2 closest pos ray 3');
        assert(*(enemy_sensors_2.at(4).mag) == 0, 'invalid v2 closest pos ray 4');
    }

    #[test]
    #[available_gas(20000000)]
    fn test_near_wall() {
        let vehicle_near_left_wall = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(CAR_WIDTH, false), FixedTrait::new(TEN, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };
        let left_wall = near_wall(vehicle_near_left_wall);
        assert(left_wall == Wall::Left(()), 'invalid near left wall');

        let vehicle_near_no_wall = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(GRID_WIDTH, false) / FixedTrait::new(TWO, false),
                FixedTrait::new(TEN, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };
        let no_wall = near_wall(vehicle_near_no_wall);
        assert(no_wall == Wall::None(()), 'invalid near no wall');

        let vehicle_near_right_wall = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(GRID_WIDTH - CAR_WIDTH, false), FixedTrait::new(TEN, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };
        let right_wall = near_wall(vehicle_near_right_wall);
        assert(right_wall == Wall::Right(()), 'invalid near right wall');
    }

    #[test]
    #[available_gas(20000000)]
    fn test_distances_to_wall() {
        // Vehicle 1 to test Wall::Left(())
        let vehicle_1 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(HUNDRED, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };
        let near_wall_1 = near_wall(vehicle_1);
        assert(near_wall_1 == Wall::Left(()), 'invalid near_wall_1');

        let ray_segments_1 = RaysTrait::new(vehicle_1.position, vehicle_1.steer).segments;
        let distances_to_wall_1 = distances_to_wall(near_wall_1, ray_segments_1);

        // Values calculated in spreadsheet "drive_ai tests"
        // Asserted values below are only for NUM_RAYS = 5
        // Vehicle 1 only ray 0 intersects with Left wall
        assert(distances_to_wall_1.len() == NUM_RAYS, 'invalid distances_to_wall_1');
        assert_precise(
            *(distances_to_wall_1.at(0)),
            2408051417826740000000,
            'invalid v1 dist to wall ray 0',
            Option::Some(184467440737095000) // 1e-02
        );
        assert(*(distances_to_wall_1.at(1).mag) == 0, 'invalid v1 dist to wall ray 1');
        assert(*(distances_to_wall_1.at(2).mag) == 0, 'invalid v1 dist to wall ray 2');
        assert(*(distances_to_wall_1.at(3).mag) == 0, 'invalid v1 dist to wall ray 3');
        assert(*(distances_to_wall_1.at(4).mag) == 0, 'invalid v1 dist to wall ray 4');

        // Vehicle 2 to test Wall::Right(())
        let vehicle_2 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(THREE_FIFTY, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(DEG_25_IN_RADS, false),
            speed: FixedTrait::new(0, false)
        };

        let near_wall_2 = near_wall(vehicle_2);
        assert(near_wall_2 == Wall::Right(()), 'invalid near_wall_2');

        let ray_segments_2 = RaysTrait::new(vehicle_2.position, vehicle_2.steer).segments;
        let distances_to_wall_2 = distances_to_wall(near_wall_2, ray_segments_2);

        // Vehicle 2 rays 2,3,4 each intersect with Right wall
        assert(distances_to_wall_2.len() == NUM_RAYS, 'invalid distances_to_wall_2');
        assert(*(distances_to_wall_2.at(0).mag) == 0, 'invalid v2 dist to wall ray 0');
        assert(*(distances_to_wall_2.at(1).mag) == 0, 'invalid v2 dist to wall ray 1');
        assert_precise(
            *(distances_to_wall_2.at(2)),
            2182435751561020000000,
            'invalid v2 dist to wall ray 2',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(distances_to_wall_2.at(3)),
            1125965820528530000000,
            'invalid v2 dist to wall ray 3',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(distances_to_wall_2.at(4)),
            954873737281615000000,
            'invalid v2 dist to wall ray 4',
            Option::Some(184467440737095000) // 1e-02
        );

        // Vehicle 3 to test Wall::None(())
        let vehicle_3 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(TWO_HUNDRED, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };

        let near_wall_3 = near_wall(vehicle_3);
        assert(near_wall_3 == Wall::None(()), 'invalid near_wall_3');

        let ray_segments_3 = RaysTrait::new(vehicle_3.position, vehicle_3.steer).segments;
        let distances_to_wall_3 = distances_to_wall(near_wall_3, ray_segments_3);

        assert(distances_to_wall_3.len() == 0, 'invalid distances_to_wall_3');
    }

    // TODO
    #[test]
    #[available_gas(20000000)]
    fn test_collision_check() {}
}
