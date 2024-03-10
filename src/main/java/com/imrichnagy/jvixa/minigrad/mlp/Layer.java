package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    List<Neuron> neurons = new ArrayList<>();
    int nNeurons;

    public Layer(int nNeurons, int nInputs, boolean useBias, Activation activation) {
        for (int i = 0; i < nNeurons; i++) {
            neurons.add(new Neuron(nInputs, useBias, activation));
        }
        this.nNeurons = nNeurons;
    }

    public List<Value> call(List<Value> inputs) {
        List<Value> output = new ArrayList<>(nNeurons);
        for (int i = 0; i < nNeurons; i++) {
            output.add(neurons.get(i).call(inputs));
        }
        return output;
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>();
        for (Neuron neuron : neurons) {
            parameters.addAll(neuron.parameters());
        }
        return parameters;
    }
}
