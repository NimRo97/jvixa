package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Value {

    public double data;
    public double gradient;

    private final Operator operator;
    private Runnable gradientFunction;
    private final Set<Value> children;
    private final String label;

    public Value(double data, Set<Value> children, Operator operator, String label) {
        this.data = data;
        this.gradient = 0;
        this.children = new HashSet<>(children);
        this.operator = operator;
        this.gradientFunction = () -> {};
        this.label = label;
    }

    public Value(double data) {
        this(data, new HashSet<>(), Operator.CONSTANT, "C" + data);
    }

    public Value(double data, String label) {
        this(data, new HashSet<>(), Operator.CONSTANT, label);
    }

    public Value add(Value other) {

        Value out = new Value(
                this.data + other.data,
                Set.of(this, other),
                Operator.ADD,
                "(" + this.label + "+" + other.label + ")"
        );

        out.gradientFunction = () -> {
            this.gradient += 1 * out.gradient;
            other.gradient += 1 * out.gradient;
        };

        return out;
    }

    public Value mul(Value other) {

        Value out = new Value(
                this.data * other.data,
                Set.of(this, other),
                Operator.MULTIPLY,
                "(" + this.label + "*" + other.label + ")"
        );

        out.gradientFunction = () -> {
            this.gradient += other.data * out.gradient;
            other.gradient += this.data * out.gradient;
        };

        return out;
    }

    public Value sub(Value other) {

        Value out = new Value(
                this.data - other.data,
                Set.of(this, other),
                Operator.SUBTRACT,
                "(" + this.label + "-" + other.label + ")"
        );

        out.gradientFunction = () -> {
            this.gradient += 1 * out.gradient;
            other.gradient += -1 * out.gradient;
        };

        return out;
    }

    public Value div(Value other) {

        Value out = new Value(
                this.data / other.data,
                Set.of(this, other),
                Operator.DIVIDE,
                "(" + this.label + "/" + other.label + ")"
        );

        out.gradientFunction = () -> {
            this.gradient += (1 / other.data) * out.gradient;
            other.gradient += (-this.data * Math.pow(other.data, -2)) * out.gradient;
        };

        return out;
    }

    public Value pow(int exp) {

        Value out = new Value(
                Math.pow(this.data, exp),
                Set.of(this),
                Operator.POWER,
                "(" + this.label + "^" + exp + ")"
        );

        out.gradientFunction = () -> this.gradient += exp * Math.pow(this.data, exp - 1) * out.gradient;

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
            v.gradientFunction.run();
        }
    }

    private void buildTopo(Value v, List<Value> topoList, Set<Value> visited) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.children) {
                buildTopo(child, topoList, visited);
            }
            topoList.add(v);
        }
    }

    @Override
    public String toString() {
        return "Value{" +
                "data=" + data +
                ", gradient=" + gradient +
                ", operator=" + operator +
                ", label='" + label + '\'' +
                '}';
    }
}
