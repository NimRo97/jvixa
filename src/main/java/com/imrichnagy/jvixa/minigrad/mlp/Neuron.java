package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class Neuron {

    List<Value> weights = new ArrayList<>();
    Value bias;
    boolean useBias;


    public Neuron(int nInputs, boolean useBias, String activation) {
        Random r = new Random();
        for (int i = 0; i < nInputs; i++) {
            weights.add(new Value(r.nextDouble(), "w"+i));
        }
        if (useBias) {
            bias = new Value(0, "b");
        }
        this.useBias = useBias;
    }

    public Value call(Value[] inputs) {
        Value out = null;
        for (int i = 0; i < inputs.length; i++) {
            if (i == 0) {
                out = inputs[i].mul(weights.get(i));
            } else {
                out = out.add(inputs[i].mul(weights.get(i)));
            }
        }
        return out;
    }

    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>(weights);
        if (useBias) {
            parameters.add(bias);
        }
        return parameters;
    }
}
