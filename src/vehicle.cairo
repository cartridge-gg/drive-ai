use array::ArrayTrait;
use cubit::types::vec2::{Vec2, Vec2Trait};
use cubit::types::fixed::{Fixed, FixedTrait, FixedPrint, ONE_u128};
use cubit::math::trig;
use cubit::math::core::neg;

#[derive(Component, Serde, Drop, Copy)]
struct Vehicle {
    // Current vehicle position
    position: Vec2,
    // Vehicle dimensions, stored in half-lengths
    length: Fixed,
    width: Fixed,
    // Vehicle steer in radians -1/2π <= s <= 1/2π
    steer: Fixed,
    // Vehicle velocity 0 <= v <= 100
    speed: Fixed
}

impl VehicleSerdeLen of dojo::SerdeLen<Vehicle> {
    #[inline(always)]
    fn len() -> usize {
        12
    }
}

#[derive(Serde, Drop)]
enum Direction {
    Straight: (),
    Left: (),
    Right: (),
}

#[derive(Serde, Drop)]
struct Controls {
    steer: Direction, 
}

// 10 degrees / pi/18 radians
const TURN_STEP: felt252 = 3219563738742341801;
const HALF_PI: felt252 = 28976077338029890953;

fn rotate(a: Vec2, sin_theta: Fixed, cos_theta: Fixed) -> Vec2 {
    // clockwise rotation is positive here
    let new_x = a.x * cos_theta + a.y * sin_theta;
    let new_y = -a.x * sin_theta + a.y * cos_theta;
    return Vec2Trait::new(new_x, new_y);
}

trait VehicleTrait {
    fn control(ref self: Vehicle, controls: Controls) -> bool;
    fn drive(ref self: Vehicle);
    fn vertices(self: @Vehicle) -> Span<Vec2>;
}

use debug::PrintTrait;

impl VehicleImpl of VehicleTrait {
    fn control(ref self: Vehicle, controls: Controls) -> bool {
        let delta = match controls.steer {
            Direction::Straight(()) => FixedTrait::from_felt(0),
            Direction::Left(()) => FixedTrait::from_felt(-1 * TURN_STEP),
            Direction::Right(()) => FixedTrait::from_felt(TURN_STEP),
        };

        // TODO: Assert bounds
        self.steer = self.steer + delta;

        (self.steer >= FixedTrait::from_felt(-1 * HALF_PI)
            && self.steer <= FixedTrait::from_felt(HALF_PI))
    }

    fn drive(ref self: Vehicle) {
        // Velocity vector
        let x_comp = self.speed * trig::sin(self.steer);
        let y_comp = self.speed * trig::cos(self.steer);
        let v_0 = Vec2Trait::new(x_comp, y_comp);

        self.position = self.position + v_0;
    }

    fn vertices(self: @Vehicle) -> Span<Vec2> {
        let mut vertices = ArrayTrait::<Vec2>::new();

        // To reduce sin and cos calculations
        let sin_theta = trig::sin_fast(*self.steer);
        let cos_theta = trig::cos_fast(*self.steer);

        let rel_vertex_0 = Vec2Trait::new(*self.width, *self.length); // relative to vehicle
        let rot_rel_vertex_0 = rotate(rel_vertex_0, sin_theta, cos_theta); // rotated rel to vehicle
        let vertex_0 = *self.position + rot_rel_vertex_0; // relative to origin

        let rel_vertex_1 = Vec2Trait::new(-*self.width, *self.length);
        let rot_rel_vertex_1 = rotate(rel_vertex_1, sin_theta, cos_theta);
        let vertex_1 = *self.position + rot_rel_vertex_1;

        // Get last two vertices by symmetry
        let vertex_2 = *self.position - rot_rel_vertex_0;
        let vertex_3 = *self.position - rot_rel_vertex_1;

        vertices.append(vertex_0);
        vertices.append(vertex_1);
        vertices.append(vertex_2);
        vertices.append(vertex_3);
        vertices.span()
    }
}

#[cfg(test)]
mod tests {
    use debug::PrintTrait;
    use cubit::types::vec2::{Vec2, Vec2Trait};
    use cubit::types::fixed::{Fixed, FixedTrait, FixedPrint};
    use cubit::test::helpers::assert_precise;
    use array::SpanTrait;

    use super::{Vehicle, VehicleTrait, Controls, Direction, TURN_STEP};

    const TEN: felt252 = 184467440737095516160;
    const TWENTY: felt252 = 368934881474191032320;
    const FORTY: felt252 = 737869762948382064640;
    const DEG_NEG_30_IN_RADS: felt252 = -9658715196994321226;

    #[test]
    #[available_gas(2000000)]
    fn test_control() {
        let mut vehicle = Vehicle {
            position: Vec2Trait::new(FixedTrait::from_felt(TEN), FixedTrait::from_felt(TEN)),
            width: FixedTrait::from_felt(TEN),
            length: FixedTrait::from_felt(TEN),
            steer: FixedTrait::new(0_u128, false),
            speed: FixedTrait::from_felt(TEN)
        };

        vehicle.control(Controls { steer: Direction::Left(()) });
        assert(vehicle.steer == FixedTrait::from_felt(-1 * TURN_STEP), 'invalid steer');
        vehicle.control(Controls { steer: Direction::Left(()) });
        assert(vehicle.steer == FixedTrait::from_felt(-2 * TURN_STEP), 'invalid steer');
        vehicle.control(Controls { steer: Direction::Right(()) });
        assert(vehicle.steer == FixedTrait::from_felt(-1 * TURN_STEP), 'invalid steer');
        vehicle.control(Controls { steer: Direction::Right(()) });
        assert(vehicle.steer == FixedTrait::from_felt(0), 'invalid steer');
    }

    #[test]
    #[available_gas(20000000)]
    fn test_drive() {
        let mut vehicle = Vehicle {
            position: Vec2Trait::new(FixedTrait::from_felt(TEN), FixedTrait::from_felt(TEN)),
            width: FixedTrait::from_felt(TEN),
            length: FixedTrait::from_felt(TEN),
            steer: FixedTrait::new(0_u128, false),
            speed: FixedTrait::from_felt(TEN)
        };

        vehicle.drive();

        assert(vehicle.position.x == FixedTrait::from_felt(TEN), 'invalid position x');
        assert(
            vehicle.position.y == FixedTrait::from_felt(368934881474199059390), 'invalid position y'
        );

        vehicle.control(Controls { steer: Direction::Left(()) });
        vehicle.drive();

        // x: ~8.263527, y: ~29.84807913671
        assert(
            vehicle.position.x == FixedTrait::from_felt(152435010392070545930), 'invalid position x'
        );
        assert(
            vehicle.position.y == FixedTrait::from_felt(550599848097669227190), 'invalid position y'
        );
    }

    #[test]
    #[available_gas(20000000)]
    fn test_vertices() {
        let mut vehicle = Vehicle {
            position: Vec2Trait::new(FixedTrait::from_felt(TEN), FixedTrait::from_felt(TWENTY)),
            width: FixedTrait::from_felt(TEN),
            length: FixedTrait::from_felt(TWENTY),
            steer: FixedTrait::new(0_u128, false),
            speed: FixedTrait::from_felt(TEN)
        };

        let mut vertices = vehicle.vertices();

        assert_precise(*(vertices.at(0).x), TWENTY, 'invalid vertex_0', Option::None(()));
        assert_precise(*(vertices.at(0).y), FORTY, 'invalid vertex_0', Option::None(()));

        assert_precise(*(vertices.at(1).x), 0, 'invalid vertex_1', Option::None(()));
        assert_precise(*(vertices.at(1).y), FORTY, 'invalid vertex_1', Option::None(()));

        assert_precise(*(vertices.at(2).x), 0, 'invalid vertex_2', Option::None(()));
        assert_precise(*(vertices.at(2).y), 0, 'invalid vertex_2', Option::None(()));

        assert_precise(*(vertices.at(3).x), TWENTY, 'invalid vertex_3', Option::None(()));
        assert_precise(*(vertices.at(3).y), 0, 'invalid vertex_3', Option::None(()));

        vehicle = Vehicle {
            position: Vec2Trait::new(FixedTrait::from_felt(TEN), FixedTrait::from_felt(TWENTY)),
            width: FixedTrait::from_felt(TEN),
            length: FixedTrait::from_felt(TWENTY),
            steer: FixedTrait::from_felt(DEG_NEG_30_IN_RADS),
            speed: FixedTrait::from_felt(TEN)
        };

        vertices = vehicle.vertices();

        // x: ~8.66025403784439, y: ~42.32050807568880
        assert_precise(
            *(vertices.at(0).x), 159753090305067335160, 'invalid rotated vertex_0', Option::None(())
        );
        assert_precise(
            *(vertices.at(0).y), 780673828410437532220, 'invalid rotated vertex_0', Option::None(())
        );
        // x: ~-8.66025403784439, y: ~32.32050807568880        
        assert_precise(
            *(vertices.at(1).x),
            -159752327071118592360,
            'invalid rotated vertex_1',
            Option::None(())
        );
        assert_precise(
            *(vertices.at(1).y), 596206769290316387460, 'invalid rotated vertex_1', Option::None(())
        );
        // x: ~11.33974596215560, y: ~-2.32050807568877
        assert_precise(
            *(vertices.at(2).x), 209181791169123697160, 'invalid rotated vertex_2', Option::None(())
        );
        assert_precise(
            *(vertices.at(2).y), -42804065462055467580, 'invalid rotated vertex_2', Option::None(())
        );
        // x: ~28.66025403784440, y: ~7.67949192431123
        assert_precise(
            *(vertices.at(3).x), 528687208545309624680, 'invalid rotated vertex_3', Option::None(())
        );
        assert_precise(
            *(vertices.at(3).y), 141662993658065677180, 'invalid rotated vertex_3', Option::None(())
        );
    }
}
