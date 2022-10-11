//import nnLUT.Cache;
//import nnLUT.Grads;
import utils.Utils;

import java.io.IOException;
import java.util.ArrayList;

public class ModelTest {

    //        double[][] test = {{0}};
//        DMatrixRMaj tm = new DMatrixRMaj(test);
//        System.out.println(nn.customSigmoidMatrixDerivative(tm));
//        System.out.println("=====================  FORWARD RESULT  =======================");
//
//        Cache c = nn.forwardPropagation(bipolarX);
//        System.out.println(c.A2);
//        System.out.println(nn.computeCost(c.A2));

//        System.out.println("=====================  BACKWARD RESULT  =======================");
//
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
//
//        nn.backwardPropagation(new Cache(null, t, null, null), binaryX, binaryY);



//    //        #################################         ##############################
//    double errorRate = 0;
//    int epoch = 1;
//    ArrayList<Double> errors = new ArrayList<Double>();
//        System.out.println("=====================  MODEL TRAINING IN PROGRESS ...  =======================");
//        do {
//        System.out.println("Training on EPOCH " + epoch++);
////            Cache cache = nn.forwardPropagation(binaryX);
//        Cache cache = nn.forwardPropagation(bipolarX);
//        errorRate = nn.computeCost(cache.A2);
//        System.out.println("Error rate for current epoch: " + errorRate);
//        errors.add(errorRate);
//        Grads grads = nn.backwardPropagation(cache, bipolarX, bipolarY);
////            Grads grads = nn.backwardPropagation(cache, binaryX, binaryY);
//
////            nn.update_parameters(grads);
//    } while (errorRate >= 0.05 && epoch <= 10000);
//
//        if (epoch >= 10000)
//            System.out.println("=====================  MODEL TRAINED FAILED :(  =======================");
//        else
//                System.out.println("=====================  MODEL TRAINED SUCCESSFULLY !  =======================");
//
//        try {
//        Utils.save("BinaryNoMomentum.csv", errors);
//    } catch (
//    IOException e) {
//        System.out.println("unable to save");
//        System.exit(1);
//    }
//
//        System.out.println("Final error rate: " + errorRate);
//}
}
