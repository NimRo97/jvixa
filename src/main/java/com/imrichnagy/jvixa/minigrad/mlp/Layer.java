package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    private final List<Neuron> neurons;


    public Layer(int neurons, int inputs, boolean useBias, Activation activation, String layerSuffix) {
        this.neurons = new ArrayList<>(neurons);
        for (int i = 0; i < neurons; i++) {
            this.neurons.add(new Neuron(inputs, useBias, activation, "N" + i, layerSuffix));
        }
    }

    public Layer(int neurons, int inputs, boolean useBias, Activation activation) {
        this(neurons, inputs, useBias, activation, null);
    }

    public List<Value> call(List<Value> inputs) {
        return neurons.stream().map(neuron -> neuron.call(inputs)).toList();
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>();
        for (Neuron neuron : neurons) {
            parameters.addAll(neuron.parameters());
        }
        return parameters;
    }
}
