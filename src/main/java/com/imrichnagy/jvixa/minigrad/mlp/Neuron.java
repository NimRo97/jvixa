package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
        this.activation = activation == null ? Activation.LINEAR : activation;
    }

    public Value call(List<Value> inputs) {

        Value out = inputs.getFirst().mul(weights.getFirst());

        for (int i = 1; i < inputs.size(); i++) {
            out = out.add(inputs.get(i).mul(weights.get(i)));
        }

        if (bias != null) {
            out = out.add(bias);
        }

        return switch (activation) {
            case LINEAR -> out;
            case SIGMOID -> out.sigmoid();
            case TANH -> out.tanh();
            case RELU -> out.relu();
        };
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>(weights);
        if (bias != null) {
            parameters.add(bias);
        }
        return parameters;
    }
}
