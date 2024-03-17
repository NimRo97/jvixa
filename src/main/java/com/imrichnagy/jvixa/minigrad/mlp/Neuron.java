package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {

    private final List<Value> weights;
    private final Value bias;
    private final Activation activation;

    private static final Random random = new Random();


    public Neuron(int inputs, boolean useBias, Activation activation, String neuronSuffix, String layerSuffix) {

        weights = new ArrayList<>(inputs);
        for (int i = 0; i < inputs; i++) {
            weights.add(new Value(
                    random.nextDouble(-1.0, 1.0),
                    "w" + i + (neuronSuffix == null ? "" : neuronSuffix) + (layerSuffix == null ? "" : layerSuffix)
            ));
        }

        bias = useBias ? new Value(0.0, "b") : null;
        this.activation = activation == null ? Activation.LINEAR : activation;
    }

    public Neuron(int inputs, boolean useBias, Activation activation) {
        this(inputs, useBias, activation, null, null);
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
