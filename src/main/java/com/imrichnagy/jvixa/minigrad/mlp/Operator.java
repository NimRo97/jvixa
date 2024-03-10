package com.imrichnagy.jvixa.minigrad.mlp;

public enum Operator {
    CONSTANT("C"),
    ADD("+"),
    MULTIPLY("*"),
    SUBTRACT("-"),
    DIVIDE("/"),
    POWER("2"),
    RELU("RELU"),
    SIGMOID("SIGMOID"),
    TANH("TANH"),
    ;

    private final String label;

    Operator(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return label;
    }
}
