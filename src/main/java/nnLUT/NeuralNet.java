package nnLUT;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.simple.SimpleMatrix;
import utils.Numva;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

class Cache {
    public DMatrixRMaj A1;
    public DMatrixRMaj A2;
    public DMatrixRMaj Z1;
    public DMatrixRMaj Z2;

    Cache(DMatrixRMaj A1, DMatrixRMaj A2, DMatrixRMaj Z1, DMatrixRMaj Z2) {
        this.A1 = A1;
        this.A2 = A2;
        this.Z1 = Z1;
        this.Z2 = Z2;
    }
}


class Grads {
    public DMatrixRMaj dW1;
    public DMatrixRMaj dW2;
    public DMatrixRMaj db1;
    public DMatrixRMaj db2;

    Grads(DMatrixRMaj dW1, DMatrixRMaj dW2, DMatrixRMaj db1, DMatrixRMaj db2) {
        this.dW1 = dW1;
        this.dW2 = dW2;
        this.db1 = db1;
        this.db2 = db2;
    }
}

public class NeuralNet implements NeuralNetInterface, CommonInterface {
    private static final int UPDATE_W1 = 1;
    private static final int UPDATE_W2 = 2;
    private static final int UPDATE_B1 = 3;
    private static final int UPDATE_B2 = 4;
    private boolean withMomentum;


    private boolean isBoolean;

    private float momentum;

    // hyper-parameters
    private int argNumHidden; // the # of neurons in hidden layer
    private int argNumOfInputs;

    private int numOfHiddenLayers = 1;
    private double argLearningRate;
    private double argMomentumTerm;
    private double outputLowerBound;
    private double outputUpperBound;
    private double weightLowerBound;
    private double weightUpperBound;

    private boolean isBatchUpdate;

//    private double[][] weights;

    public DMatrixRMaj W1;
    public DMatrixRMaj W2;
    public DMatrixRMaj b1;
    public DMatrixRMaj b2;
    private DMatrixRMaj input;
    private DMatrixRMaj expectedOutput;

    private int m;

    private Map<String, DMatrixRMaj> velocity;


    // constants
    private final double ERROR_THRESHOLD = 0.05;
    private final int INPUT_DIM = 2;

    public NeuralNet (
            int argNumOfInputs,
            int argNumHidden,
            int numOfHiddenLayers,
            double argLearningRate,
            double argMomentumTerm,
            double outputLowerBound,
            double outputUpperBound,
            double weightLowerBound,
            double weightUpperBound,
            boolean withMomentum,
            boolean isBatchUpdate,
            DMatrixRMaj input,
            DMatrixRMaj expectedOutput) {

        this.argNumOfInputs = argNumOfInputs;
        this.argNumHidden = argNumHidden;
        this.numOfHiddenLayers = numOfHiddenLayers;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.outputLowerBound = outputLowerBound;
        this.outputUpperBound = outputUpperBound;
        this.weightLowerBound = weightLowerBound;
        this.weightUpperBound = weightUpperBound;
        this.input = input;
        this.expectedOutput = expectedOutput;
        this.m = input.numCols;
        this.withMomentum = withMomentum;
        this.isBatchUpdate = isBatchUpdate;

        this.velocity = new HashMap<>();
        this.initializeWeights();
        this.initializeBias();
        this.initializeVelocity();
    }

    private void initializeVelocity() {
        velocity.put("dW1", new DMatrixRMaj(this.W1.numRows, this.W1.numCols));
        velocity.put("dW2", new DMatrixRMaj(this.W2.numRows, this.W2.numCols));
        velocity.put("db1", new DMatrixRMaj(this.b1.numRows, this.b1.numCols));
        velocity.put("db2", new DMatrixRMaj(this.b2.numRows, this.b2.numCols));
    }


    /**
     * Return a bipolar sigmoid of the input X
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */

    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.pow(Math.E, -x)) - 1;

    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param X The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    public DMatrixRMaj sigmoidMatrix(DMatrixRMaj X) {
        DMatrixRMaj res = new DMatrixRMaj(X.numRows, X.numCols);
        CommonOps_DDRM.scale(-1d, X);  // -X
        CommonOps_DDRM.elementPower(Math.E, X, res); // e^(-x)
        CommonOps_DDRM.add(res, 1d); //(1+ e^(-x))
        CommonOps_DDRM.divide(res, 1d); // 1 / (1 + e^(-x))
        return res;
    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param X The input
     * @return f(x) = b_minus_a / (1 + e(-X)) - minus_a
     */

    public DMatrixRMaj customSigmoidMatrix(DMatrixRMaj X) {
        DMatrixRMaj Xrep = new DMatrixRMaj(X);
        DMatrixRMaj res = new DMatrixRMaj(Xrep.numRows, Xrep.numCols);
        CommonOps_DDRM.scale(-1d, Xrep);
        CommonOps_DDRM.elementPower(Math.E, Xrep, res);
        CommonOps_DDRM.add(res, 1d);
        CommonOps_DDRM.divide(this.outputUpperBound - this.outputLowerBound, res);
        CommonOps_DDRM.add(res, this.outputLowerBound);

        return res;
//        return X;
    }

    public DMatrixRMaj customSigmoidMatrixDerivative(DMatrixRMaj X) {



        DMatrixRMaj Xrep = new DMatrixRMaj(X);
        DMatrixRMaj numerator = new DMatrixRMaj(Xrep.numRows, Xrep.numCols);
        CommonOps_DDRM.scale(-1d, Xrep);
        CommonOps_DDRM.elementPower(Math.E, Xrep, numerator);

        DMatrixRMaj t = new DMatrixRMaj(numerator);
        CommonOps_DDRM.add(t, 1d);

        DMatrixRMaj denominator = new DMatrixRMaj(t.numRows, t.numCols);
        CommonOps_DDRM.elementMult(t, t, denominator);

        DMatrixRMaj res = new DMatrixRMaj(numerator.numRows, numerator.numCols);
        CommonOps_DDRM.elementDiv(numerator, denominator, res);

        DMatrixRMaj constantMx = new DMatrixRMaj(res.numRows, res.numCols);
        CommonOps_DDRM.fill(constantMx,this.outputUpperBound - this.outputLowerBound);

        DMatrixRMaj gPrime = new DMatrixRMaj(res.numRows, res.numCols);
        CommonOps_DDRM.elementMult(constantMx, res, gPrime);


        return gPrime;
    }




    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */

    @Override
    public double customSigmoid(double x) {
        return (outputUpperBound - outputLowerBound) / (1 + Math.pow(Math.E, -x)) + outputLowerBound;
    }



    @Override
    public void initializeWeights() {

        this.W1 = new DMatrixRMaj(this.argNumHidden, this.argNumOfInputs);
        this.W2 = new DMatrixRMaj(1, this.argNumHidden);
        RandomMatrices_DDRM.fillUniform(this.W1, this.weightLowerBound, this.weightUpperBound, new Random());
        RandomMatrices_DDRM.fillUniform(this.W2, this.weightLowerBound, this.weightUpperBound, new Random());
    }

    @Override
    public void zeroWeights() {
        this.W1 = new DMatrixRMaj(this.argNumHidden, this.argNumOfInputs);
        this.W2 = new DMatrixRMaj(1, this.argNumHidden);
    }

    public void oneWeights() {
        this.W1 = new DMatrixRMaj(this.argNumHidden, this.argNumOfInputs);
        this.W2 = new DMatrixRMaj(1, this.argNumHidden);

        CommonOps_DDRM.fill(W1, 1d);
        CommonOps_DDRM.fill(W2, 1d);
    }

    public Cache forwardPropagation(DMatrixRMaj X) {
        DMatrixRMaj Z1 = new DMatrixRMaj(this.W1.numRows, X.numCols);
        CommonOps_DDRM.mult(this.W1, X, Z1);

        CommonOps_DDRM.add(Z1, this.reformatBiasMatrix(this.b1, Z1.numCols), Z1);

        DMatrixRMaj A1 = customSigmoidMatrix(Z1);


        DMatrixRMaj Z2 = new DMatrixRMaj(this.W2.numRows, A1.numCols);
        CommonOps_DDRM.mult(this.W2, A1, Z2);


        CommonOps_DDRM.add(Z2, this.reformatBiasMatrix(this.b2, Z2.numCols), Z2);
        DMatrixRMaj A2 = customSigmoidMatrix(Z2);

        Cache cache = new Cache(A1, A2, Z1, Z2);

        return cache;
    }

    private DMatrixRMaj reformatBiasMatrix(DMatrixRMaj bias, int times) {
        DMatrixRMaj t = new DMatrixRMaj(bias);
        for (int i = 0; i < times - 1; ++i) {
            t = CommonOps_DDRM.concatColumnsMulti(t, bias);
        }

        return t;
    }

    public double computeCost(DMatrixRMaj A2) {
        DMatrixRMaj res = new DMatrixRMaj(A2.numRows, A2.numCols);

        CommonOps_DDRM.subtract(this.expectedOutput, A2, res);
        DMatrixRMaj t = new DMatrixRMaj(res.numRows, res.numCols);
        DMatrixRMaj res2 = new DMatrixRMaj(res);
        CommonOps_DDRM.transpose(res2);

        CommonOps_DDRM.mult(res, res2, t);

        double cost = CommonOps_DDRM.elementSum(t);
        return 0.5 * cost;
    }

    public Grads backwardPropagation(Cache cache, DMatrixRMaj X, DMatrixRMaj Y) {

        DMatrixRMaj A1 = cache.A1;
        DMatrixRMaj A2 = cache.A2;
        DMatrixRMaj Z1 = cache.Z1;
        DMatrixRMaj Z2 = cache.Z2;

        // dA2 = A2 - Y
        DMatrixRMaj dA2 = new DMatrixRMaj(A2.numRows, A2.numCols);
        CommonOps_DDRM.subtract(A2, Y, dA2);

//        DMatrixRMaj dA2dZ2 = this.customSigmoidMatrixDerivative(A2);
        DMatrixRMaj dA2dZ2 = this.customSigmoidMatrixDerivative(Z2);



        DMatrixRMaj db2 = new DMatrixRMaj(dA2.numRows, dA2dZ2.numCols);
        CommonOps_DDRM.elementMult(dA2, dA2dZ2, db2);

        DMatrixRMaj dZ2 = new DMatrixRMaj(db2);

        DMatrixRMaj db2Compressed = new DMatrixRMaj(this.b2.numRows, this.b2.numCols);
        CommonOps_DDRM.sumRows(db2, db2Compressed);
//        CommonOps_DDRM.divide(db2Compressed, m);
        if (!isBatchUpdate) {
            if (this.withMomentum)
                this.update_parameter_momentum(db2Compressed, UPDATE_B2);
            else
                this.update_parameter(db2Compressed, UPDATE_B2);
        }


        DMatrixRMaj A1Trans = new DMatrixRMaj(A1);
        CommonOps_DDRM.transpose(A1Trans);
        DMatrixRMaj dW2 = new DMatrixRMaj(db2.numRows, A1Trans.numCols);
        CommonOps_DDRM.mult(dZ2, A1Trans, dW2);
//        CommonOps_DDRM.divide(dW2, m);

        if (!this.isBatchUpdate) {
            if (this.withMomentum)
                this.update_parameter_momentum(dW2, UPDATE_W2);
            else
                this.update_parameter(dW2, UPDATE_W2);
        }

        DMatrixRMaj t2 = new DMatrixRMaj(db2.numRows, W2.numCols);
        DMatrixRMaj W2Trans = new DMatrixRMaj(W2);
        CommonOps_DDRM.transpose(W2Trans);
        CommonOps_DDRM.mult(W2Trans, dZ2, t2);
//        DMatrixRMaj sigmoidZ1 = customSigmoidMatrixDerivative(A1);

        DMatrixRMaj sigmoidZ1 = customSigmoidMatrixDerivative(Z1);
        DMatrixRMaj db1 = new DMatrixRMaj(t2.numRows, sigmoidZ1.numCols);
//        DMatrixRMaj db1 = new DMatrixRMaj(t2.numRows, Z1.numCols);

        CommonOps_DDRM.elementMult(t2, sigmoidZ1, db1);

        DMatrixRMaj dZ1 = new DMatrixRMaj(db1);

        DMatrixRMaj db1Compressed = new DMatrixRMaj(b1.numRows, b1.numCols);
        CommonOps_DDRM.sumRows(db1, db1Compressed);
//        CommonOps_DDRM.divide(db1Compressed, m);
        if (!this.isBatchUpdate) {
            if (this.withMomentum)
                this.update_parameter_momentum(db1Compressed, UPDATE_B1);
            else
                this.update_parameter(db1Compressed, UPDATE_B1);
        }

        DMatrixRMaj XTrans = new DMatrixRMaj(X);
        CommonOps_DDRM.transpose(XTrans);
        DMatrixRMaj dW1 = new DMatrixRMaj(db1.numRows, XTrans.numCols);
        CommonOps_DDRM.mult(dZ1, XTrans, dW1);
//        CommonOps_DDRM.divide(dW1, m);
        if (!this.isBatchUpdate) {
            if (this.withMomentum)
                this.update_parameter_momentum(dW1, UPDATE_W1);
            else
                this.update_parameter(dW1, UPDATE_W1);
        }



//        DMatrixRMaj db1Compressed = new DMatrixRMaj(b1.numRows, b1.numCols);
//        DMatrixRMaj db2Compressed = new DMatrixRMaj(b2.numRows, b2.numCols);
//        CommonOps_DDRM.sumRows(db1, db1Compressed);
//        CommonOps_DDRM.sumRows(db2, db2Compressed);
//        CommonOps_DDRM.divide(db1Compressed, m);
//        CommonOps_DDRM.divide(db2Compressed, m)
        return new Grads(dW1, dW2, db1Compressed, db2Compressed);
    }

    public void initializeBias() {
        this.b1 = new DMatrixRMaj(W1.numRows, 1);
        this.b2 = new DMatrixRMaj(W2.numRows, 1);
        CommonOps_DDRM.fill(b1, NeuralNet.bias);
        CommonOps_DDRM.fill(b2, NeuralNet.bias);
    }

    public void update_parameter(DMatrixRMaj derivative, int term) {
        DMatrixRMaj lrMatrix;
        DMatrixRMaj t;


        // i = i - argLearningRate * di;

        lrMatrix = new DMatrixRMaj(derivative.numRows, derivative.numCols);
        CommonOps_DDRM.fill(lrMatrix, argLearningRate);
        t = new DMatrixRMaj(derivative.numRows, derivative.numCols);
        CommonOps_DDRM.elementMult(derivative, lrMatrix, t);
        switch (term){
            case UPDATE_W1 -> {
                CommonOps_DDRM.subtract(this.W1, t, this.W1);
            }
            case UPDATE_W2 -> {
                CommonOps_DDRM.subtract(this.W2, t, this.W2);
            }
            case UPDATE_B1 -> {
                CommonOps_DDRM.subtract(this.b1, t, this.b1);
            }
            case UPDATE_B2 -> {
                CommonOps_DDRM.subtract(this.b2, t, this.b2);
            }
            default -> System.out.println("Please only update W1, W2, B1, B2");
        }
    }

    public void update_parameter_momentum(DMatrixRMaj derivative, int term) {

        switch (term){
            case UPDATE_W1 -> {
                // v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
                DMatrixRMaj v = this.velocity.get("dW1");
                DMatrixRMaj temp = new DMatrixRMaj(v);
                CommonOps_DDRM.divide(v, 1 / this.argMomentumTerm, temp);
                DMatrixRMaj temp2 = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.divide(derivative, 1 / (1 - this.argMomentumTerm), temp2);

                DMatrixRMaj newV = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.add(temp, temp2, newV);
                this.velocity.put("dW1", newV);

                DMatrixRMaj newV2 = new DMatrixRMaj(newV.numRows, newV.numCols);
                CommonOps_DDRM.divide(newV, this.argLearningRate, newV2);
                CommonOps_DDRM.subtract(this.W1, newV2, this.W1);
            }
            case UPDATE_W2 -> {
                DMatrixRMaj v = this.velocity.get("dW2");
                DMatrixRMaj temp = new DMatrixRMaj(v);
                CommonOps_DDRM.divide(v, 1 / this.argMomentumTerm, temp);
                DMatrixRMaj temp2 = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.divide(derivative, 1 / (1 - this.argMomentumTerm), temp2);

                DMatrixRMaj newV = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.add(temp, temp2, newV);
                this.velocity.put("dW2", newV);

                DMatrixRMaj newV2 = new DMatrixRMaj(newV.numRows, newV.numCols);
                CommonOps_DDRM.divide(newV, this.argLearningRate, newV2);
                CommonOps_DDRM.subtract(this.W2, newV2, this.W2);
            }
            case UPDATE_B1 -> {
                DMatrixRMaj v = this.velocity.get("db1");
                DMatrixRMaj temp = new DMatrixRMaj(v);
                CommonOps_DDRM.divide(v, 1 / this.argMomentumTerm, temp);
                DMatrixRMaj temp2 = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.divide(derivative, 1 / (1 - this.argMomentumTerm), temp2);

                DMatrixRMaj newV = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.add(temp, temp2, newV);
                this.velocity.put("db1", newV);

                DMatrixRMaj newV2 = new DMatrixRMaj(newV.numRows, newV.numCols);
                CommonOps_DDRM.divide(newV, this.argLearningRate, newV2);
                CommonOps_DDRM.subtract(this.b1, newV2, this.b1);
            }
            case UPDATE_B2 -> {
                DMatrixRMaj v = this.velocity.get("db2");
                DMatrixRMaj temp = new DMatrixRMaj(v);
                CommonOps_DDRM.divide(v, 1 / this.argMomentumTerm, temp);
                DMatrixRMaj temp2 = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.divide(derivative, 1 / (1 - this.argMomentumTerm), temp2);

                DMatrixRMaj newV = new DMatrixRMaj(derivative.numRows, derivative.numCols);
                CommonOps_DDRM.add(temp, temp2, newV);
                this.velocity.put("db2", newV);

                DMatrixRMaj newV2 = new DMatrixRMaj(newV.numRows, newV.numCols);
                CommonOps_DDRM.divide(newV, this.argLearningRate, newV2);
                CommonOps_DDRM.subtract(this.b2, newV2, this.b2);
            }
            default -> System.out.println("Please only update W1, W2, B1, B2");
        }
    }

    public void batch_update(Grads grads) {
        DMatrixRMaj dW1 = grads.dW1;
        DMatrixRMaj dW2 = grads.dW2;
        DMatrixRMaj db1 = grads.db1;
        DMatrixRMaj db2 = grads.db2;
        DMatrixRMaj lrMatrix;
        DMatrixRMaj t;


        // b2 =  b2 - argLearningRate * db2;
        lrMatrix = new DMatrixRMaj(db2.numRows, db2.numCols);
        CommonOps_DDRM.fill(lrMatrix, argLearningRate);
        t = new DMatrixRMaj(db2.numRows, db2.numCols);
        CommonOps_DDRM.elementMult(db2, lrMatrix, t);
        CommonOps_DDRM.subtract(this.b2, t, this.b2);


        // this.W2 = W2 - argLearningRate * dW2;
        lrMatrix = new DMatrixRMaj(dW2.numRows, dW2.numCols);
        CommonOps_DDRM.fill(lrMatrix, argLearningRate);
        t = new DMatrixRMaj(this.W2.numRows, this.W2.numCols);
        CommonOps_DDRM.elementMult(dW2, lrMatrix, t);
        CommonOps_DDRM.subtract(this.W2, t, this.W2);


        // b1 = b1 - argLearningRate * db1;
        lrMatrix = new DMatrixRMaj(db1.numRows, db1.numCols);
        CommonOps_DDRM.fill(lrMatrix, argLearningRate);
        t = new DMatrixRMaj(db1.numRows, db1.numCols);
        CommonOps_DDRM.elementMult(db1, lrMatrix, t);
        CommonOps_DDRM.subtract(this.b1, t, this.b1);

        // this.W1 = W1 - argLearningRate * dW1;
        lrMatrix = new DMatrixRMaj(dW1.numRows, dW1.numCols);
        CommonOps_DDRM.fill(lrMatrix, argLearningRate);
        t = new DMatrixRMaj(this.W1.numRows, this.W1.numCols);
        CommonOps_DDRM.elementMult(dW1, lrMatrix, t);
        CommonOps_DDRM.subtract(this.W1, t, this.W1);
    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public ArrayList<Double> train() {
        double errorRate = 0;
        int epoch = 1;
        ArrayList<Double> errors = new ArrayList<Double>();
        System.out.println("=====================  MODEL TRAINING IN PROGRESS ...  =======================");
        do {
            System.out.println("Training on EPOCH " + epoch++);
            Cache cache = this.forwardPropagation(this.input);
            errorRate = this.computeCost(cache.A2);
            System.out.println("Error rate for current epoch: " + errorRate);
            errors.add(errorRate);
            Grads grads = this.backwardPropagation(cache, this.input, this.expectedOutput);
            if (this.isBatchUpdate)
                this.batch_update(grads);
        } while (errorRate >= NeuralNetInterface.errorThreshHold);

        System.out.println("=====================  MODEL TRAINED SUCCESSFULLY !  =======================");


        return errors;
    }

    @Override
    public void save(String fileName, ArrayList<Double> errors) {
        try {
            Utils.save(fileName, errors);
        } catch (IOException e) {
            System.out.println("unable to save");
            System.exit(1);
        }
    }

    @Override
    public void load(String argFileName) throws IOException {

    }
}
