package com.imrichnagy.jvixa.minigrad.mlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

public class Value {

    public double data;
    public double gradient;

    private final Operator operator;
    private Runnable gradientFunction;
    private final Set<Value> children;
    private final String label;

    public Value(double data, String label, Operator operator, Value... children) {
        this.data = data;
        this.gradient = 0;
        this.children = Set.of(children);
        this.operator = operator;
        this.gradientFunction = () -> {};
        this.label = label;
    }

    public Value(double data, String label) {
        this(data, label, Operator.CONSTANT);
    }

    public Value(double data) {
        this(data, "C" + data, Operator.CONSTANT);
    }

    public Value add(Value other) {

        Value out = new Value(
                this.data + other.data,
                "(" + this.label + "+" + other.label + ")", Operator.ADD, this, other
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
                "(" + this.label + "*" + other.label + ")", Operator.MULTIPLY, this, other
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
                "(" + this.label + "-" + other.label + ")", Operator.SUBTRACT, this, other
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
                "(" + this.label + "/" + other.label + ")", Operator.DIVIDE, this, other
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
                "(" + this.label + "^" + exp + ")", Operator.POWER, this
        );

        out.gradientFunction = () -> this.gradient += exp * Math.pow(this.data, exp - 1) * out.gradient;

        return out;
    }

    public Value sigmoid() {
        return this; // TODO
    }

    public Value tanh() {
        return this; // TODO
    }

    public Value relu() {
        return this; // TODO
    }

    public void backward() {
        Set<Value> visited = new HashSet<>();
        List<Value> topography = new ArrayList<>();

        buildTopography(topography, visited);
        this.gradient = 1;

        // reverse iteration
        ListIterator<Value> listIterator = topography.listIterator(topography.size());
        while (listIterator.hasPrevious()) {
            listIterator.previous().gradientFunction.run();
        }
    }

    private void buildTopography(List<Value> topography, Set<Value> visited) {

        if (visited.contains(this)) {
            return;
        }

        visited.add(this);
        for (Value child : children) {
            child.buildTopography(topography, visited);
        }
        topography.add(this);
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
