package nnLUT;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import utils.Utils;

import java.io.IOException;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        double[][] binaryInput = {
                {1d, 0d, 1d, 0d},
                {0d, 1d, 1d, 0d}
        };

        double[][] binaryOutput =  {{
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



        NeuralNet nn = new NeuralNet(
                2,
                4,
                1,
                0.5,
                0.0,
                0,
                1.0,
                -0.5,
                0.5,
                binaryX,
                binaryY
                );

        nn.initializeWeights();
//        nn.oneWeights();
        nn.initializeBias();


//        System.out.println("=====================  FORWARD RESULT  =======================");

//        Cache cache = nn.forwardPropagation(binaryX);
//        System.out.println(cache.A2);
//        System.out.println(nn.computeCost(cache.A2));

//        System.out.println("=====================  BACKWARD RESULT  =======================");

//        Grads grad = nn.backwardPropagation(cache, binaryX, binaryY);
//        System.out.println(grad.dW1);
//        System.out.println(grad.dW2);
//        System.out.println(grad.db1);
//        System.out.println(grad.db2);


//        double[][] A2 = {
//                {1,1,0,1}
//        };
//
//        DMatrixRMaj t = new DMatrixRMaj(A2);
//        System.out.println(nn.customSigmoidMatrix(t));
//        System.out.print(nn.customSigmoidMatrixDerivative(t));

//        nn.backwardPropagation(new Cache(null, t, null, null), binaryX, binaryY);
        double errorRate = 0;
        int epoch = 1;
        ArrayList<Double> errors = new ArrayList<Double>();
        System.out.println("=====================  MODEL TRAINING IN PROGRESS ...  =======================");
        do {
            System.out.println("Training on EPOCH " + epoch++);
            Cache cache = nn.forwardPropagation(binaryX);
            errorRate = nn.computeCost(cache.A2);
            System.out.println("Error rate for current epoch: " + errorRate);
            errors.add(errorRate);
            Grads grads = nn.backwardPropagation(cache, binaryX, binaryY);
//            nn.update_parameters(grads);
        } while (errorRate >= 0.05 && epoch <= 10000);

        if (epoch >= 10000)
            System.out.println("=====================  MODEL TRAINED FAILED :(  =======================");
        else
            System.out.println("=====================  MODEL TRAINED SUCCESSFULLY !  =======================");

        try {
            Utils.save("BinaryNoMomentum.txt", errors);
        } catch (IOException e) {
            System.out.println("unable to save");
            System.exit(1);
        }

        System.out.println("Final error rate: " + errorRate);

    }
}