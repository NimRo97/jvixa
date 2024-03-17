package com.imrichnagy.jvixa.minigrad;

import com.imrichnagy.jvixa.minigrad.mlp.Activation;
import com.imrichnagy.jvixa.minigrad.mlp.Neuron;
import com.imrichnagy.jvixa.minigrad.mlp.Value;

import java.io.IOException;
import java.util.List;

import static com.imrichnagy.jvixa.minigrad.GraphVisualizer.visualize;

public class VisualizerPlayground {
    public static void main(String[] args) throws IOException {
        Neuron neuron = new Neuron(2, false, Activation.LINEAR);
        Value out = neuron.call(List.of(new Value(1), new Value(2)));
        visualize(out, "out");
    }
}
