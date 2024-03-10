package com.imrichnagy.jvixa;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Network {
    private List<Layer> layers = new ArrayList<>();

    public Network(int nInputs, int[] nLayers, boolean useBias, String activation) {
        for (int i = 0; i < nLayers.length; i++) {

            int inputs;
            if (i == 0) {
                inputs = nInputs;
            } else {
                inputs = nLayers[i-1];
            }

            layers.add(new Layer(nLayers[i], inputs, useBias, activation));
        }
    }

    public Value[] call(Value[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.call(inputs);
        }
        return inputs;
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    public void resetGradients() {
        for (Value p : parameters()) {
            p.gradient = 0;
        }
    }

    public void update(double descent) {
        List<Value> parameters = this.parameters();
        for (Value param: parameters) {
            param.data += -descent * param.gradient;
        }
    }
}