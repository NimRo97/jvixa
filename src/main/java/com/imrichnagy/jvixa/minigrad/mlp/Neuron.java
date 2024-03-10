package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class Neuron {

    List<Value> weights;
    Value bias = null;
    Activation activation;

    private static final Random random = new Random();


    public Neuron(int inputs, boolean useBias, Activation activation) {
        weights = new ArrayList<>(inputs);
        for (int i = 0; i < inputs; i++) {
            weights.add(new Value(random.nextDouble(-1, 1), "w" + i));
        }
        if (useBias) {
            bias = new Value(0, "b");
        }
        this.activation = activation;
    }

    public Value call(List<Value> inputs) {
        Value out = null;
        for (int i = 0; i < inputs.size(); i++) {
            if (i == 0) {
                out = inputs.get(i).mul(weights.get(i));
            } else {
                out = out.add(inputs.get(i).mul(weights.get(i)));
            }
        }
        return out;
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>(weights);
        if (bias != null) {
            parameters.add(bias);
        }
        return parameters;
    }
}
