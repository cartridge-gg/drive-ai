#[system]
mod model {
    use array::ArrayTrait;
    use dojo::world::Context;
    use drive_ai::racer::Sensors;
    use drive_ai::vehicle::{Controls, Direction};
    use orion::operators::nn::core::NNTrait;
    use orion::operators::nn::implementations::impl_nn_i8::NN_i8;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
    use orion::numbers::signed_integer::i8::i8;
    use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
    use orion::performance::core::PerfomanceTrait;
    use orion::performance::implementations::impl_performance_fp::Performance_fp_i8;
    use orion::operators::nn::functional::linear::linear_ft::linear_ft;
    use core::traits::Into;
    use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE, FP8x23Mul};
    fn execute(ctx: Context, sensors: Sensors) -> Controls {
        let prediction: usize = forward(sensors.rays);
        let steer: Direction = if prediction == 0 {
            Direction::Straight(())
        } else if prediction == 1 {
            Direction::Left(())
        } else if prediction == 2 {
            Direction::Right(())
        } else {
            let mut panic_msg = ArrayTrait::new();
            panic_msg.append('prediction must be < 3');
            panic(panic_msg)
        };
        Controls { steer: steer,  }
    }
    fn forward(input: Tensor<FixedType>) -> usize {
        let w = fc1::fc1_weights();
        let b = fc1::fc1_bias();
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(1, false));
        let extra = Option::<ExtraParams>::None(());
        let y_scale = TensorTrait::new(shape.span(), data.span(), extra);
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(0, false));
        let extra = Option::<ExtraParams>::None(());
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), extra);
        let x = linear_ft(input, w, b);
        *x.argmax(0, Option::None(()), Option::None(())).data[0]
    }
    mod fc0 {
        use array::ArrayTrait;
        use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::signed_integer::i8::i8;
        use orion::numbers::fixed_point::core::FixedType;
        use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
        fn fc0_weights() -> Tensor<FixedType> {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(3);
            shape.append(3);
            let mut data = ArrayTrait::<FixedType>::new();
            data.append(FixedType { mag: 1527555, sign: true });
            data.append(FixedType { mag: 6880309, sign: true });
            data.append(FixedType { mag: 2046218, sign: false });
            data.append(FixedType { mag: 3959852, sign: true });
            data.append(FixedType { mag: 164384, sign: true });
            data.append(FixedType { mag: 11918, sign: false });
            data.append(FixedType { mag: 3840494, sign: true });
            data.append(FixedType { mag: 4174026, sign: true });
            data.append(FixedType { mag: 5626131, sign: true });
            let extra = Option::<ExtraParams>::None(());
            TensorTrait::new(shape.span(), data.span(), extra)
        }
        fn fc0_bias() -> Tensor<FixedType> {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(3);
            let mut data = ArrayTrait::<FixedType>::new();
            data.append(FixedType { mag: 0, sign: false });
            data.append(FixedType { mag: 0, sign: false });
            data.append(FixedType { mag: 0, sign: false });
            let extra = Option::<ExtraParams>::None(());
            TensorTrait::new(shape.span(), data.span(), extra)
        }
    }
    mod fc1 {
        use array::ArrayTrait;
        use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::signed_integer::i8::i8;
        use orion::numbers::fixed_point::core::FixedType;
        use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
        fn fc1_weights() -> Tensor<FixedType> {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(1);
            shape.append(4);
            let mut data = ArrayTrait::<FixedType>::new();
            data.append(FixedType { mag: 2179121, sign: false });
            data.append(FixedType { mag: 4536469, sign: true });
            data.append(FixedType { mag: 3153609, sign: true });
            data.append(FixedType { mag: 2399109, sign: true });
            let extra = Option::<ExtraParams>::None(());
            TensorTrait::new(shape.span(), data.span(), extra)
        }
        fn fc1_bias() -> Tensor<FixedType> {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(3);
            let mut data = ArrayTrait::<FixedType>::new();
            data.append(FixedType { mag: 0, sign: false });
            data.append(FixedType { mag: 0, sign: false });
            data.append(FixedType { mag: 0, sign: false });
            let extra = Option::<ExtraParams>::None(());
            TensorTrait::new(shape.span(), data.span(), extra)
        }
    }
}
