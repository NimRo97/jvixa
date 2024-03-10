package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Value {

    public double data;
    public double gradient;
    private final String operator;

    private final Set<Value> _prev;

    private Runnable localgradients;

    private final String label;

    public Value(double data, Set<Value> _children, String _operator, String label) {
        this.data = data;
        this.gradient = 0;
        this._prev = new HashSet<>(_children);
        this.operator = _operator;
        this.localgradients = () -> {
        };
        this.label = label;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, Set.of(this, other), "+", this.label + "+" + other.label);
        out.localgradients = () -> {
            this.gradient += 1 * out.gradient;
            other.gradient += 1 * out.gradient;
        };

        return out;
    }

    public Value mul(Value other) {
        Value out = new Value(this.data * other.data, Set.of(this, other), "*", this.label + "*" + other.label);
        out.localgradients = () -> {
            this.gradient += other.data * out.gradient;
            other.gradient += this.data * out.gradient;
        };

        return out;
    }

    public Value sub(Value other) {
        Value out = new Value(this.data - other.data, Set.of(this, other), "-", this.label + "-" + other.label);
        out.localgradients = () -> {
            this.gradient += 1 * out.gradient;
            other.gradient += - this.data * out.gradient;
        };

        return out;
    }

    public Value div(Value other) {
        Value out = new Value(this.data / other.data, Set.of(this, other), "/", this.label + "/" + other.label);
        out.localgradients = () -> {
            this.gradient += (1 / other.data) * out.gradient;
            other.gradient += (-this.data * Math.pow(other.data, -2)) * out.gradient;
        };

        return out;
    }

    public Value pow(int exp) {
        Value out = new Value(Math.pow(this.data, exp), Set.of(this), "^(" + exp + ")", this.label + "power");
        out.localgradients = () -> this.gradient += exp * Math.pow(this.data, exp - 1) * out.gradient;

        return out;
    }

    public void backward() {
        Set<Value> topo = new HashSet<>();
        Set<Value> visited = new HashSet<>();
        List<Value> topoList = new ArrayList<>(topo);

        buildTopo(this, topoList, visited);
        this.gradient = 1;

        for (int i = topoList.size() - 1; i >= 0; i--) {
            Value v = topoList.get(i);
            v.localgradients.run();
        }
    }

    private void buildTopo(Value v, List<Value> topoList, Set<Value> visited) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v._prev) {
                buildTopo(child, topoList, visited);
            }
            topoList.add(v);
        }
    }
}
