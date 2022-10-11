package utils;

import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

public class Main {
    public static void main(String[] args) {
//        System.out.println(Numva.getRandomInt(-1, 1));
//        System.out.println(Numva.getRandomFloat((float) -0.5, (float) 0.5));
        SimpleMatrix firstMatrix = new SimpleMatrix(
                new double[][] {
                        new double[] {1, 5},
                        new double[] {2, 3},
                        new double[] {1 ,7}
                }
        );

        SimpleMatrix secondMatrix = new SimpleMatrix(
                new double[][] {
                        new double[] {1, 2, 3, 7},
                        new double[] {5, 2, 8, 1}
                }
        );


        SimpleMatrix expected = new SimpleMatrix(
                new double[][] {
                        new double[] {26, 12, 43, 12},
                        new double[] {17, 10, 30, 17},
                        new double[] {36, 16, 59, 14}
                }
        );

        SimpleMatrix actual = firstMatrix.mult(secondMatrix);
        System.out.println(actual);
//        Equation eq = new Equation();
//        System.out.println(eq.dot())
    }
}
