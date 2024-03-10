package com.imrichnagy.jvixa;

import java.util.HashSet;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        int trainingCycles = 5000;
        int[] hiddenLayers = {10, 10, 1};
        int batchSize = 100;
        int terms = 10;
        int range = 100;
        double descend = 1.4e-7;
        Random random = new Random();


        //training data
        Value[][] in = makeData(terms, batchSize, range, random);
        Network network = new Network(terms, hiddenLayers, false, null);
        Value count = new Value(batchSize, new HashSet<>(), "c", "c");

        for (int i = 0; i < trainingCycles; i++) {
            Value loss = null;
            //forward pass

            for (int b = 0; b < batchSize; b++) {
                Value[] out = network.call(in[b]);
                Value pred = fun(in[b]);

                if (b == 0) {
                    loss = out[0].sub(pred).pow(2);
                } else {
                    loss = loss.add(out[0].sub(pred).pow(2));
                }
            }
            loss=loss.div(count);

            //reset gradients
            network.resetGradients();

            //backward pass
            loss.backward();

            //update parameters
            network.update(descend);

            System.out.println("Iteration " + i + " | mse: " + loss.data);
        }

        in = makeData(terms, 10, range, random);

        for (int b = 0; b < 10; b++) {
            Value[] out = network.call(in[b]);
            String addition = "";
            for (int i = 0; i < in[b].length; i++) {
                addition += in[b][i].data;
                if (i < in[b].length - 1) {
                    addition += " + ";
                }
            }
            System.out.println(addition + " = " + out[0].data);
        }
    }


    public static Value[][] makeData(int length, int range, int batchSize, Random random) {

        double sum = 0;
        Value[][] batch = new Value[batchSize][length];
        for (int b = 0; b < batchSize; b++) {
            Value[] inputs = new Value[length];
            for (int i = 0; i < length; i++) {
                double val = (double) random.nextInt(range);
                inputs[i] = new Value(val, new HashSet<>(), "X", "x" + i);
            }
            batch[b] = inputs;
        }
        return batch;
    }

    public static Value fun(Value[] inputs) {
        double sum = 0;
        for (Value input : inputs) {
            sum += input.data;
        }
        Value out = new Value(sum, new HashSet<>(), "Y", "Y");
        return out;
    }
}