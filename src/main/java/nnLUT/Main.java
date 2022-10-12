package nnLUT;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import utils.Utils;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        double[][] binaryInput = {
                {1d, 0d, 1d, 0d},
                {0d, 1d, 1d, 0d}
        };

        double[][] binaryOutput = {{
                1d,
                1d,                                                    /*Output Vector binary*/
                0d,
                0d
        }};

        double[][] bipolarOutput = {{
                -1d,
                1d,                                                    /*Output Vector Bipolar*/
                1d,
                -1d
        }};

        double[][] bipolarInput = {
                {-1d, -1d, 1d, 1d},
                {-1d, 1d, -1d, 1d}
        };

        DMatrixRMaj binaryX = new DMatrixRMaj(binaryInput);
        DMatrixRMaj bipolarX = new DMatrixRMaj(bipolarInput);
        DMatrixRMaj binaryY = new DMatrixRMaj(binaryOutput);
        DMatrixRMaj bipolarY = new DMatrixRMaj(bipolarOutput);


        NeuralNet binaryNeuralNetWithMomentum = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.9,
                0.0,
                1.0,
                -0.5,
                0.5,
                true,
                binaryX,
                binaryY
                );

        NeuralNet binaryNeuralNetNoMomentum = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.0,
                0.0,
                1.0,
                -0.5,
                0.5,
                false,
                binaryX,
                binaryY
        );

        NeuralNet bipolarNeuralNetWithMomentum = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.9,
                -1.0,
                1.0,
                -0.5,
                0.5,
                true,
                bipolarX,
                bipolarY
        );

        NeuralNet bipolarNeuralNetNoMomentum = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.0,
                -1.0,
                1.0,
                -0.5,
                0.5,
                false,
                bipolarX,
                bipolarY
        );

//        binaryNeuralNet.initializeWeights();
//        binaryNeuralNet.initializeBias();
//        bipolarNeuralNet.initializeWeights();
//        bipolarNeuralNet.initializeBias();

        ArrayList<Double> errors;
        errors = binaryNeuralNetWithMomentum.train();
        binaryNeuralNetWithMomentum.save("BinaryWithMomentum.csv", errors);

        errors = binaryNeuralNetNoMomentum.train();
        binaryNeuralNetWithMomentum.save("BinaryWithNoMomentum.csv", errors);

        errors = bipolarNeuralNetWithMomentum.train();
        bipolarNeuralNetWithMomentum.save("BipolarWithMomentum.csv", errors);

        errors = bipolarNeuralNetNoMomentum.train();
        bipolarNeuralNetNoMomentum.save("BipolarWithNoMomentum.csv", errors);

    }
}