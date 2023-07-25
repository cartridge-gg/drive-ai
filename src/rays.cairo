use cubit::math::trig;
use cubit::types::vec2::{Vec2, Vec2Trait};
use cubit::types::fixed::{Fixed, FixedTrait};
use array::{ArrayTrait, SpanTrait};

use drive_ai::math::{distance, intersects};

const DEG_90_IN_RADS: u128 = 28976077338029890953;
const DEG_70_IN_RADS: u128 = 22536387234850959209;
const DEG_50_IN_RADS: u128 = 16098473553126325695;
const DEG_30_IN_RADS: u128 = 9658715196994321226;
const DEG_10_IN_RADS: u128 = 3218956840862316756;

const NUM_RAYS: usize = 5;
const RAY_LENGTH: u128 = 2767011611056432742400; // 150

#[derive(Serde, Drop)]
struct Rays {
    segments: Span<Ray>, 
}

trait RaysTrait {
    fn new(position: Vec2, theta: Fixed) -> Rays;
}

impl RaysImpl of RaysTrait {
    fn new(position: Vec2, theta: Fixed) -> Rays {
        let ray_length = FixedTrait::new(RAY_LENGTH, false);

        let mut rays_theta = ArrayTrait::new();
        // rays_theta.append(theta + FixedTrait::new(DEG_70_IN_RADS, true));
        rays_theta.append(theta + FixedTrait::new(DEG_50_IN_RADS, true));
        rays_theta.append(theta + FixedTrait::new(DEG_30_IN_RADS, true));
        // rays_theta.append(theta + FixedTrait::new(DEG_10_IN_RADS, true));
        rays_theta.append(theta);
        // rays_theta.append(theta + FixedTrait::new(DEG_10_IN_RADS, false));
        rays_theta.append(theta + FixedTrait::new(DEG_30_IN_RADS, false));
        rays_theta.append(theta + FixedTrait::new(DEG_50_IN_RADS, false));
        // rays_theta.append(theta + FixedTrait::new(DEG_70_IN_RADS, false));

        // TODO: Rays are semetric, we calculate half and invert
        let mut segments = ArrayTrait::new();
        loop {
            match rays_theta.pop_front() {
                Option::Some(theta) => {
                    // Endpoints of Ray
                    // TODO: Rays are semetric, we calculate half and invert
                    let cos_theta = trig::cos_fast(theta);
                    let sin_theta = trig::sin_fast(theta);
                    let delta1 = Vec2Trait::new(ray_length * sin_theta, ray_length * cos_theta);

                    // TODO: We currently project out the center point?
                    let q = position + delta1;

                    segments.append(Ray { theta, cos_theta, sin_theta, p: position, q,  });
                },
                Option::None(_) => {
                    break ();
                }
            };
        };

        Rays { segments: segments.span() }
    }
}

#[derive(Serde, Drop)]
struct Ray {
    theta: Fixed,
    cos_theta: Fixed,
    sin_theta: Fixed,
    p: Vec2,
    q: Vec2,
}

trait RayTrait {
    fn intersects(self: @Ray, p: Vec2, q: Vec2) -> bool;
    fn dist(self: @Ray, p: Vec2, q: Vec2) -> Fixed;
}

impl RayImpl of RayTrait {
    fn intersects(self: @Ray, p: Vec2, q: Vec2) -> bool {
        intersects(*self.p, *self.q, p, q)
    }
    fn dist(self: @Ray, p: Vec2, q: Vec2) -> Fixed {
        distance(*self.p, p, q, *self.cos_theta, *self.sin_theta)
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
    use drive_ai::enemy::{Position, PositionTrait};
    use drive_ai::racer::{
        compute_sensors, filter_positions, closest_position, near_wall, distances_to_wall,
        collision_check, Wall
    };
    use drive_ai::racer::{GRID_HEIGHT, GRID_WIDTH, CAR_HEIGHT, CAR_WIDTH};
    use super::{Rays, RaysTrait, Ray, RayTrait, RAY_LENGTH};

    const NUM_RAYS: usize = 5; // Asserted values below are only for NUM_RAYS = 5
    const TWO: u128 = 36893488147419103232;
    const FOUR: u128 = 73786976294838206464;
    const TEN: u128 = 184467440737095516160;
    const FIFTY: u128 = 922337203685477580800;
    const HUNDRED: u128 = 1844674407370955161600;
    const TWO_HUNDRED: u128 = 3689348814741910323200;
    const THREE_HUNDRED: u128 = 5534023222112865484800;
    const THREE_FIFTY: u128 = 6456360425798343065600;
    const DEG_25_IN_RADS: u128 = 8048910508974580935;

    #[test]
    #[available_gas(20000000)]
    fn test_raystrait_new() {
        // Vehicle 1
        let vehicle_1 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(HUNDRED, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(0, false),
            speed: FixedTrait::new(0, false)
        };
        let ray_segments_1 = RaysTrait::new(vehicle_1.position, vehicle_1.steer).segments;
        assert(ray_segments_1.len() == NUM_RAYS, 'invalid ray_segments_1');

        // Values calculated in spreadsheet "drive_ai tests"
        // Asserted values below are only for NUM_RAYS = 5
        // ray_segments_1.at(0)
        assert_precise(
            *(ray_segments_1.at(0).theta),
            -16097821017949100000,
            'invalid ray_segments_1 0 theta',
            Option::Some(1844674407370950) // use custom_precision of 1e-04
        );
        assert_precise(
            *(ray_segments_1.at(0).cos_theta),
            11857338529639100000,
            'invalid ray_segments_1 0 cos',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_1.at(0).sin_theta),
            -14131025791303100000,
            'invalid ray_segments_1 0 sin',
            Option::Some(1844674407370950) // 1e-04
        );
        // p & q are less precise due to propagation of error above
        assert_precise(
            *(ray_segments_1.at(0).p.x),
            1844674407370950000000,
            'invalid ray_segments_1 0 p.x',
            // 
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_1.at(0).p.y),
            3689348814741900000000,
            'invalid ray_segments_1 0 p.y',
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_1.at(0).q.x),
            -274979461324515000000,
            'invalid ray_segments_1 0 q.x',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(ray_segments_1.at(0).q.y),
            5467949594187760000000,
            'invalid ray_segments_1 0 q.y',
            Option::Some(184467440737095000) // 1e-02
        );
        // ray_segments_1.at(3)
        assert_precise(
            *(ray_segments_1.at(3).theta),
            9658692610769470000,
            'invalid ray_segments_1 3 theta',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_1.at(3).cos_theta),
            15975348984942500000,
            'invalid ray_segments_1 3 cos',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_1.at(3).sin_theta),
            9223372036854750000,
            'invalid ray_segments_1 3 sin',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_1.at(3).p.x),
            1844674407370950000000,
            'invalid ray_segments_1 3 p.x',
            // 
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_1.at(3).p.y),
            3689348814741900000000,
            'invalid ray_segments_1 3 p.y',
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_1.at(3).q.x),
            3228180212899160000000,
            'invalid ray_segments_1 3 q.x',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(ray_segments_1.at(3).q.y),
            6085651162483270000000,
            'invalid ray_segments_1 3 q.y',
            Option::Some(184467440737095000) // 1e-02
        );

        // Vehicle 2
        let vehicle_2 = Vehicle {
            position: Vec2Trait::new(
                FixedTrait::new(THREE_FIFTY, false), FixedTrait::new(TWO_HUNDRED, false)
            ),
            steer: FixedTrait::new(DEG_25_IN_RADS, false),
            speed: FixedTrait::new(0, false)
        };
        let ray_segments_2 = RaysTrait::new(vehicle_2.position, vehicle_2.steer).segments;
        assert(ray_segments_2.len() == NUM_RAYS, 'invalid ray_segments_2');

        // ray_segments_2.at(0)
        assert_precise(
            *(ray_segments_2.at(0).theta),
            -8048910508974560000,
            'invalid ray_segments_2 0 theta',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(0).cos_theta),
            16718427799475100000,
            'invalid ray_segments_2 0 cos',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(0).sin_theta),
            -7795930915206660000,
            'invalid ray_segments_2 0 sin',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(0).p.x),
            6456360425798330000000,
            'invalid ray_segments_2 0 p.x',
            // 
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_2.at(0).p.y),
            3689348814741900000000,
            'invalid ray_segments_2 0 p.y',
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_2.at(0).q.x),
            5286970788517330000000,
            'invalid ray_segments_2 0 q.x',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(ray_segments_2.at(0).q.y),
            6197112984663160000000,
            'invalid ray_segments_2 0 q.y',
            Option::Some(184467440737095000) // 1e-02
        );
        // ray_segments_2.at(3)
        assert_precise(
            *(ray_segments_2.at(3).theta),
            17707603119744000000,
            'invalid ray_segments_2 3 theta',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(3).cos_theta),
            10580617728078100000,
            'invalid ray_segments_2 3 cos',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(3).sin_theta),
            15110688118455000000,
            'invalid ray_segments_2 3 sin',
            Option::Some(1844674407370950) // 1e-04
        );
        assert_precise(
            *(ray_segments_2.at(3).p.x),
            6456360425798330000000,
            'invalid ray_segments_2 3 p.x',
            // 
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_2.at(3).p.y),
            3689348814741900000000,
            'invalid ray_segments_2 3 p.y',
            Option::Some(18446744073709500) // 1e-03
        );
        assert_precise(
            *(ray_segments_2.at(3).q.x),
            8722963643566570000000,
            'invalid ray_segments_2 3 q.x',
            Option::Some(184467440737095000) // 1e-02
        );
        assert_precise(
            *(ray_segments_2.at(3).q.y),
            5276441473953610000000,
            'invalid ray_segments_2 3 q.y',
            Option::Some(184467440737095000) // 1e-02
        );
    }

    #[test]
    #[available_gas(20000000)]
    fn test_raytrait_intersects() {
        // // From vehicle 1 above, rays 0, 1, & 3 intersect enemy edges
        // // Ray 0
        // let ray_10 = Ray {
        //     theta: FixedTrait::new(16097821017949100000, true),
        //     cos_theta: FixedTrait::new(11857338529639100000, false),
        //     sin_theta: FixedTrait::new(14131025791303100000, true),
        //     p: Vec2Trait::new(
        //         FixedTrait::new(1844674407370950000000, false),
        //         FixedTrait::new(3689348814741900000000, false)
        //     ),
        //     q: Vec2Trait::new(
        //         FixedTrait::new(274979461324515000000, true),
        //         FixedTrait::new(5467949594187760000000, false)
        //     ),
        // };

        // // ray_10 intersects only enemy 0 edges 1 & 2
        // // Enemy 0 edge 0 (horizontal, top)
        // let p_00 = Vec2Trait::new(
        //     FixedTrait::new(885443715538056000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let q_00 = Vec2Trait::new(
        //     FixedTrait::new(295147905179352000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let intersect_1000 = ray_10.intersects(p_00, q_00);
        // assert(!intersect_1000, 'invalid RayTrait intersect 1000');
        // // Enemy 0 edge 1 (vertical, left)
        // let p_01 = Vec2Trait::new(
        //     FixedTrait::new(295147905179352000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let q_01 = Vec2Trait::new(
        //     FixedTrait::new(295147905179352000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let intersect_1001 = ray_10.intersects(p_01, q_01);
        // assert(intersect_1001, 'invalid RayTrait intersect 1001');
        // // Enemy 0 edge 2 (horizontal, bottom)
        // let p_02 = Vec2Trait::new(
        //     FixedTrait::new(295147905179352000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let q_02 = Vec2Trait::new(
        //     FixedTrait::new(885443715538056000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let intersect_1002 = ray_10.intersects(p_02, q_02);
        // assert(intersect_1002, 'invalid RayTrait intersect 1002');
        // // Enemy 0 edge 3 (vertical, right)
        // let p_03 = Vec2Trait::new(
        //     FixedTrait::new(885443715538056000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let q_03 = Vec2Trait::new(
        //     FixedTrait::new(885443715538056000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let intersect_1003 = ray_10.intersects(p_03, q_03);
        // assert(!intersect_1003, 'invalid RayTrait intersect 1003');

        // // Ray 1
        // let ray_11 = Ray {
        //     theta: FixedTrait::new(9658692610769470000, true),
        //     cos_theta: FixedTrait::new(15975348984942500000, false),
        //     sin_theta: FixedTrait::new(9223372036854750000, true),
        //     p: Vec2Trait::new(
        //         FixedTrait::new(1844674407370950000000, false),
        //         FixedTrait::new(3689348814741900000000, false)
        //     ),
        //     q: Vec2Trait::new(
        //         FixedTrait::new(461168601842738000000, false),
        //         FixedTrait::new(6085651162483270000000, false)
        //     ),
        // };

        // // ray_11 intersects only enemy 0 edge 3
        // let intersect_1100 = ray_11.intersects(p_00, q_00);
        // assert(!intersect_1100, 'invalid RayTrait intersect 1100');
        // let intersect_1101 = ray_11.intersects(p_01, q_01);
        // assert(!intersect_1101, 'invalid RayTrait intersect 1101');
        // let intersect_1102 = ray_11.intersects(p_02, q_02);
        // assert(!intersect_1102, 'invalid RayTrait intersect 1102');
        // let intersect_1103 = ray_11.intersects(p_03, q_03);
        // assert(intersect_1103, 'invalid RayTrait intersect 1103');

        // // Ray 3
        // let ray_13 = Ray {
        //     theta: FixedTrait::new(9658692610769470000, false),
        //     cos_theta: FixedTrait::new(15975348984942500000, false),
        //     sin_theta: FixedTrait::new(9223372036854750000, false),
        //     p: Vec2Trait::new(
        //         FixedTrait::new(1844674407370950000000, false),
        //         FixedTrait::new(3689348814741900000000, false)
        //     ),
        //     q: Vec2Trait::new(
        //         FixedTrait::new(3228180212899160000000, false),
        //         FixedTrait::new(6085651162483270000000, false)
        //     ),
        // };

        // // ray_13 intersects only enemy 1 edges 2 & 3
        // // Enemy 1 edge 0 (horizontal, bottom)
        // let p_10 = Vec2Trait::new(
        //     FixedTrait::new(2951479051793520000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let q_10 = Vec2Trait::new(
        //     FixedTrait::new(2361183241434820000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let intersect_1310 = ray_13.intersects(p_10, q_10);
        // assert(!intersect_1310, 'invalid RayTrait intersect_1310');
        // // Enemy 1 edge 1 (vertical, left)
        // let p_11 = Vec2Trait::new(
        //     FixedTrait::new(2361183241434820000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let q_11 = Vec2Trait::new(
        //     FixedTrait::new(2361183241434820000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let intersect_1311 = ray_13.intersects(p_11, q_11);
        // assert(!intersect_1311, 'invalid RayTrait intersect 1311');
        // // Enemy 1 edge 2 (horizontal, bottom)
        // let p_12 = Vec2Trait::new(
        //     FixedTrait::new(2361183241434820000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let q_12 = Vec2Trait::new(
        //     FixedTrait::new(2951479051793520000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let intersect_1312 = ray_13.intersects(p_12, q_12);
        // assert(intersect_1312, 'invalid RayTrait intersect_1312');
        // // Enemy 1 edge 3 (vertical, right)
        // let p_13 = Vec2Trait::new(
        //     FixedTrait::new(2951479051793520000000, false),
        //     FixedTrait::new(4943727411754150000000, false)
        // );
        // let q_13 = Vec2Trait::new(
        //     FixedTrait::new(2951479051793520000000, false),
        //     FixedTrait::new(6124319032471550000000, false)
        // );
        // let intersect_1313 = ray_13.intersects(p_13, q_13);
        // assert(intersect_1313, 'invalid RayTrait intersect 1313');

        // From vehicle 2 above, only ray 2 intersects enemy edges
        let ray_22 = Ray {
            theta: FixedTrait::new(8048910508974560000, false),
            cos_theta: FixedTrait::new(16718427799475100000, false),
            sin_theta: FixedTrait::new(7795930915206660000, false),
            p: Vec2Trait::new(
                FixedTrait::new(6456360425798330000000, false),
                FixedTrait::new(3689348814741900000000, false)
            ),
            q: Vec2Trait::new(
                FixedTrait::new(7625750063079320000000, false),
                FixedTrait::new(6197112984663160000000, false)
            ),
        };

        // ray_22 intersects only enemy 3 edges 2 & 3
        // Enemy 3 edge 0 (horizontal, top)
        let p_30 = Vec2Trait::new(
            FixedTrait::new(6493253913945740000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let q_30 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let intersect_2230 = ray_22.intersects(p_30, q_30);
        assert(!intersect_2230, 'invalid RayTrait intersect_2230');
        // Enemy 3 edge 1 (vertical, left)
        let p_31 = Vec2Trait::new(
            FixedTrait::new(6493253913945740000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_31 = Vec2Trait::new(
            FixedTrait::new(6493253913945740000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let intersect_2231 = ray_22.intersects(p_31, q_31);
        assert(!intersect_2231, 'invalid RayTrait intersect_2231');
        // Enemy 3 edge 2 (horizontal, bottom)
        let p_32 = Vec2Trait::new(
            FixedTrait::new(6493253913945740000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_32 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let intersect_2232 = ray_22.intersects(p_32, q_32);
        assert(intersect_2232, 'invalid RayTrait intersect_2232');
        // Enemy 3 edge 3 (vertical, right)
        let p_33 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_33 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let intersect_2233 = ray_22.intersects(p_33, q_33);
        assert(intersect_2233, 'invalid RayTrait intersect_2233');
    }

    #[test]
    #[available_gas(20000000)]
    fn test_raytrait_dist() {
        // From vehicle 1 above, rays 0, 1, & 3 intersect enemy edges
        // Vehicle 1 ray 0
        let ray_10 = Ray {
            theta: FixedTrait::new(16097821017949100000, true),
            cos_theta: FixedTrait::new(11857338529639100000, false),
            sin_theta: FixedTrait::new(14131025791303100000, true),
            p: Vec2Trait::new(
                FixedTrait::new(1844674407370950000000, false),
                FixedTrait::new(3689348814741900000000, false)
            ),
            q: Vec2Trait::new(
                FixedTrait::new(274979461324515000000, true),
                FixedTrait::new(5467949594187760000000, false)
            ),
        };

        // ray_10 intersects only enemy 0 edges 1 & 2
        // Enemy 0 edge 1 (vertical, left)
        let p_01 = Vec2Trait::new(
            FixedTrait::new(295147905179352000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let q_01 = Vec2Trait::new(
            FixedTrait::new(295147905179352000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let distance_1001 = ray_10.dist(p_01, q_01);
        assert_precise(
            distance_1001,
            2022763190974460000000,
            'invalid RayTrait distance 1001',
            Option::None(())
        );
        // Enemy 0 edge 2 (horizontal, bottom)
        let p_02 = Vec2Trait::new(
            FixedTrait::new(295147905179352000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_02 = Vec2Trait::new(
            FixedTrait::new(885443715538056000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let distance_1002 = ray_10.dist(p_02, q_02);
        assert_precise(
            distance_1002,
            1951466671275690000000,
            'invalid RayTrait distance 1002',
            Option::None(())
        );

        // Vehicle 1 ray 1
        let ray_11 = Ray {
            theta: FixedTrait::new(9658692610769470000, true),
            cos_theta: FixedTrait::new(15975348984942500000, false),
            sin_theta: FixedTrait::new(9223372036854750000, true),
            p: Vec2Trait::new(
                FixedTrait::new(1844674407370950000000, false),
                FixedTrait::new(3689348814741900000000, false)
            ),
            q: Vec2Trait::new(
                FixedTrait::new(461168601842738000000, false),
                FixedTrait::new(6085651162483270000000, false)
            ),
        };

        // ray_11 intersects only enemy 0 edge 3
        // Enemy 0 edge 3 (vertical, right)
        let p_03 = Vec2Trait::new(
            FixedTrait::new(885443715538056000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_03 = Vec2Trait::new(
            FixedTrait::new(885443715538056000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let distance_1103 = ray_11.dist(p_03, q_03);
        assert_precise(
            distance_1103,
            1918461383665790000000,
            'invalid RayTrait distance 1103',
            Option::None(())
        );

        // Vehicle 1 ray 3
        let ray_13 = Ray {
            theta: FixedTrait::new(9658692610769470000, false),
            cos_theta: FixedTrait::new(15975348984942500000, false),
            sin_theta: FixedTrait::new(9223372036854750000, false),
            p: Vec2Trait::new(
                FixedTrait::new(1844674407370950000000, false),
                FixedTrait::new(3689348814741900000000, false)
            ),
            q: Vec2Trait::new(
                FixedTrait::new(3228180212899160000000, false),
                FixedTrait::new(6085651162483270000000, false)
            ),
        };

        // ray_13 intersects only enemy 1 edges 2 & 3
        // Enemy 1 edge 2 (horizontal, bottom)
        let p_12 = Vec2Trait::new(
            FixedTrait::new(2361183241434820000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_12 = Vec2Trait::new(
            FixedTrait::new(2951479051793520000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let distance_1312 = ray_11.dist(p_12, q_12);
        assert_precise(
            distance_1312,
            1448431641301450000000,
            'invalid RayTrait distance 1312',
            Option::None(())
        );

        // Enemy 1 edge 3 (vertical, right)
        let p_13 = Vec2Trait::new(
            FixedTrait::new(2951479051793520000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_13 = Vec2Trait::new(
            FixedTrait::new(2951479051793520000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let distance_1313 = ray_11.dist(p_13, q_13);
        assert_precise(
            distance_1313,
            2213609288845140000000,
            'invalid RayTrait distance 1313',
            Option::None(())
        );

        // From vehicle 2 above, only ray 2 intersects enemy edges
        // Vehicle 2 ray 2
        let ray_22 = Ray {
            theta: FixedTrait::new(8048910508974560000, false),
            cos_theta: FixedTrait::new(16718427799475100000, false),
            sin_theta: FixedTrait::new(7795930915206660000, false),
            p: Vec2Trait::new(
                FixedTrait::new(6456360425798330000000, false),
                FixedTrait::new(3689348814741900000000, false)
            ),
            q: Vec2Trait::new(
                FixedTrait::new(7625750063079320000000, false),
                FixedTrait::new(6197112984663160000000, false)
            ),
        };

        // ray_22 intersects only enemy 3 edges 2 & 3
        // Enemy 3 edge 2 (horizontal, bottom)
        let p_32 = Vec2Trait::new(
            FixedTrait::new(6493253913945740000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_32 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let distance_2232 = ray_22.dist(p_32, q_32);
        assert_precise(
            distance_2232, 1384053645962460000000, 'invalid RayTrait dist 2232', Option::None(())
        );

        // Enemy 3 edge 3 (vertical, right)
        let p_33 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(4943727411754150000000, false)
        );
        let q_33 = Vec2Trait::new(
            FixedTrait::new(7083549724304450000000, false),
            FixedTrait::new(6124319032471550000000, false)
        );
        let distance_2233 = ray_22.dist(p_33, q_33);
        assert_precise(
            distance_2233, 1484056311061490000000, 'invalid RayTrait dist 2233', Option::None(())
        );
    }
}
