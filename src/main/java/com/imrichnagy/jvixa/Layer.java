package com.imrichnagy.jvixa;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    List<Neuron> neurons = new ArrayList<>();
    int nNeurons;

    public Layer(int nNeurons, int nInputs, boolean useBias, String activation) {
        for (int i = 0; i < nNeurons; i++) {
            neurons.add(new Neuron(nInputs, useBias, activation));
        }
        this.nNeurons = nNeurons;
    }

    public Value[] call(Value[] inputs) {
        Value[] output = new Value[nNeurons];
        for (int i = 0; i < nNeurons; i++) {
            output[i] = neurons.get(i).call(inputs);
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
