package com.imrichnagy.jvixa.minigrad;

import com.imrichnagy.jvixa.minigrad.mlp.Activation;
import com.imrichnagy.jvixa.minigrad.mlp.Network;
import com.imrichnagy.jvixa.minigrad.mlp.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TrainingGrounds {
    public static void main(String[] args) {
        int trainingCycles = 5000;
        int[] hiddenLayers = {10, 10, 1};
        int batchSize = 100;
        int terms = 10;
        int range = 100;
        double descend = 1.8e-6; // 1.7e-6 + no bias for LINEAR
        Random random = new Random();


        //training data
        List<List<Value>> in = makeTestData(terms, batchSize, range, random);
        Network network = new Network(terms, false, Activation.LINEAR, hiddenLayers);
        Value count = new Value(batchSize);

        for (int i = 0; i < trainingCycles; i++) {
            Value loss = null;
            //forward pass

            for (int b = 0; b < batchSize; b++) {
                List<Value> out = network.call(in.get(b));
                Value expected = fun(in.get(b));

                if (b == 0) {
                    loss = out.getFirst().sub(expected).pow(2);
                } else {
                    loss = loss.add(out.getFirst().sub(expected).pow(2));
                }
            }
            loss=loss.div(count);

            //reset gradients
            network.resetGradients();

            //backward pass
            loss.backward();

            //update parameters
            network.update(descend);

            if (i % 50 == 0) {
                System.out.println("Iteration " + i + " | mse: " + loss.data);
            }
        }

        in = makeTestData(terms, 10, range, random);

        for (int b = 0; b < 10; b++) {
            List<Value> out = network.call(in.get(b));
            String addition = "";
            for (int i = 0; i < in.get(b).size(); i++) {
                addition += in.get(b).get(i).data;
                if (i < in.get(b).size() - 1) {
                    addition += " + ";
                }
            }
            System.out.println(addition + " = " + out.getFirst().data);
        }
    }


    private static List<List<Value>> makeTestData(int length, int range, int batchSize, Random random) {

        List<List<Value>> batch = new ArrayList<>(batchSize);
        for (int b = 0; b < batchSize; b++) {
            List<Value> inputs = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                double val = random.nextInt(range);
                inputs.add(new Value(val, "x" + i + "(" + val + ")"));
            }
            batch.add(inputs);
        }
        return batch;
    }

    private static Value fun(List<Value> inputs) {
        double sum = 0;
        for (Value input : inputs) {
            sum += input.data;
        }
        return new Value(sum);
    }
}