package com.imrichnagy.jvixa.minigrad;

import com.imrichnagy.jvixa.minigrad.mlp.Network;
import com.imrichnagy.jvixa.minigrad.mlp.Operator;
import com.imrichnagy.jvixa.minigrad.mlp.Value;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class TrainingGrounds {
    public static void main(String[] args) {
        int trainingCycles = 5000;
        int[] hiddenLayers = {10, 10, 1};
        int batchSize = 100;
        int terms = 10;
        int range = 100;
        double descend = 1.4e-7;
        Random random = new Random();


        //training data
        List<List<Value>> in = makeData(terms, batchSize, range, random);
        Network network = new Network(terms, hiddenLayers, false, null);
        Value count = new Value(batchSize);

        for (int i = 0; i < trainingCycles; i++) {
            Value loss = null;
            //forward pass

            for (int b = 0; b < batchSize; b++) {
                List<Value> out = network.call(in.get(b));
                Value pred = fun(in.get(b));

                if (b == 0) {
                    loss = out.getFirst().sub(pred).pow(2);
                } else {
                    loss = loss.add(out.getFirst().sub(pred).pow(2));
                }
            }
            loss=loss.div(count);

            //reset gradients
            network.resetGradients();

            //backward pass
            loss.backward();

            //update parameters
            network.update(descend);

            if (i % 10 == 0) {
                System.out.println("Iteration " + i + " | mse: " + loss.data);
            }
        }

        in = makeData(terms, 10, range, random);

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


    public static List<List<Value>> makeData(int length, int range, int batchSize, Random random) {

        double sum = 0;
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

    public static Value fun(List<Value> inputs) {
        double sum = 0;
        for (Value input : inputs) {
            sum += input.data;
        }
        Value out = new Value(sum);
        return out;
    }
}