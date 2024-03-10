package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.List;

public class Network {

    private final List<Layer> layers;


    public Network(int inputs, boolean useBias, Activation activation, int... layers) {

        this.layers = new ArrayList<>(layers.length);
        this.layers.add(new Layer(layers[0], inputs, useBias, activation));

        for (int i = 1; i < layers.length; i++) {
            this.layers.add(new Layer(layers[i], layers[i-1], useBias, activation));
        }
    }

    public List<Value> call(List<Value> inputs) {
        for (Layer layer : layers) {
            inputs = layer.call(inputs);
        }
        return inputs;
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>();
        for (Layer layer : layers) {
            parameters.addAll(layer.parameters());
        }
        return parameters;
    }

    public void resetGradients() {
        for (Value parameter : parameters()) {
            parameter.gradient = 0.0;
        }
    }

    public void update(double descent) {
        for (Value parameter: parameters()) {
            parameter.data += -descent * parameter.gradient;
        }
    }
}
