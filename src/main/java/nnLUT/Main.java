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


        NeuralNet binaryNeuralNet = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.0,
                0.0,
                1.0,
                -0.5,
                0.5,
                binaryX,
                binaryY
                );

        NeuralNet bipolarNeuralNet = new NeuralNet(
                2,
                4,
                1,
                0.2,
                0.0,
                -1.0,
                1.0,
                -0.5,
                0.5,
                bipolarX,
                bipolarY
        );

        binaryNeuralNet.initializeWeights();
        binaryNeuralNet.initializeBias();
        bipolarNeuralNet.initializeWeights();
        bipolarNeuralNet.initializeBias();

        ArrayList<Double> errors;
        errors = binaryNeuralNet.train();
        binaryNeuralNet.save("BinaryNoMomentum.csv", errors);
    }
}