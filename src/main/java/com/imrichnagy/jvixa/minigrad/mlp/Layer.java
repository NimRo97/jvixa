package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    private final List<Neuron> neurons;

    public Layer(int neurons, int inputs, boolean useBias, Activation activation) {
        this.neurons = new ArrayList<>(neurons);
        for (int i = 0; i < neurons; i++) {
            this.neurons.add(new Neuron(inputs, useBias, activation));
        }
    }

    public List<Value> call(List<Value> inputs) {
        return neurons.stream().map(neuron -> neuron.call(inputs)).toList();
    }

    public List<Value> parameters() {
        List<Value> list = new ArrayList<>();
        for (Neuron neuron : neurons) {
            list.addAll(neuron.parameters());
        }
        return list;
    }
}
