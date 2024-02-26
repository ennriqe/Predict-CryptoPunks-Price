use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::nn::{NNTrait, FP16x16NN};

use node_layer1_weight::get_node_layer1_weight;
use node_layer1_bias::get_node_layer1_bias;
use node_layer2_weight::get_node_layer2_weight;
use node_layer2_bias::get_node_layer2_bias;
use node_output_layer_weight::get_node_output_layer_weight;
use node_output_layer_bias::get_node_output_layer_bias;


fn main(node_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
let node__layer1_gemm_output_0 = NNTrait::gemm(node_input, get_node_layer1_weight(), Option::Some(get_node_layer1_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node__relu_relu_output_0 = NNTrait::relu(@node__layer1_gemm_output_0);
let node__layer2_gemm_output_0 = NNTrait::gemm(node__relu_relu_output_0, get_node_layer2_weight(), Option::Some(get_node_layer2_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node__relu_1_relu_output_0 = NNTrait::relu(@node__layer2_gemm_output_0);
let node_output = NNTrait::gemm(node__relu_1_relu_output_0, get_node_output_layer_weight(), Option::Some(get_node_output_layer_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);

        node_output
    }