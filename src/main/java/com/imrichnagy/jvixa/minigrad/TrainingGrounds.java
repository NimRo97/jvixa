package com.imrichnagy.jvixa.minigrad;

import com.imrichnagy.jvixa.minigrad.mlp.Activation;
import com.imrichnagy.jvixa.minigrad.mlp.Network;
import com.imrichnagy.jvixa.minigrad.mlp.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TrainingGrounds {

    private static final Random random = new Random();

    public static void main(String[] args) {

        int trainingCycles = 2500;
        int updateFrequency = 50;
        int[] hiddenLayers = {10, 10, 1};
        int batches = 100;
        int terms = 10;
        int range = 10;
        double descend = 1.6e-4;

        //training data
        List<List<Value>> trainingData = makeTestData(terms, range, batches);
        Network network = new Network(terms, false, Activation.LINEAR, hiddenLayers);

        for (int i = 0; i < trainingCycles; i++) {

            //forward pass
            Value loss = new Value(0);
            for (List<Value> batch : trainingData) {
                Value out = network.call(batch).getFirst();
                Value expected = fun(batch);

                loss = loss.add(out.sub(expected).pow(2));
            }
            loss=loss.div(new Value(batches));

            //reset gradients
            network.resetGradients();

            //backward pass
            loss.backward();

            //update parameters
            network.update(descend);

            if (i % updateFrequency == 0) {
                System.out.println("Iteration " + i + " | mse: " + loss.data);
            }
        }

        List<List<Value>> testData = makeTestData(terms, range, 10);
        for (List<Value> batch : testData) {
            Value out = network.call(batch).getFirst();
            String addition = String.join(" + ", batch.stream().map(v -> "" + v.data).toList());
            System.out.println(addition + " = " + out.data);
        }
    }

    private static List<List<Value>> makeTestData(int length, int range, int batchSize) {

        List<List<Value>> batch = new ArrayList<>(batchSize);

        for (int b = 0; b < batchSize; b++) {
            List<Value> inputs = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                double val = random.nextInt(range);
                inputs.add(new Value(val, "x" + i));
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
