package com.imrichnagy.jvixa.minigram.jbigram;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NativeNgram {

    public static final List<String> NAME_ALPHABET = List.of(
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            ":", ";"
    );

    private final List<String> alphabet;
    private final long[][] frequencies;
    private final Random random = new Random();

    public NativeNgram(List<String> alphabet, long[][] frequencies) {
        this.alphabet = alphabet;
        this.frequencies = frequencies;
    }

    public Character getNextCharacter(String current) {
        long[] freq = frequencies[getPosition(current)];
        long sum = Arrays.stream(freq).sum();
        long r = random.nextLong(sum);
        for (int i = 0; i < freq.length; i++) {
            r -= freq[i];
            if (r <= 0) {
                return getCharacter(i);
            }
        }
        return getCharacter(freq.length - 1);
    }

    private int getPosition(String s) {
        return alphabet.indexOf(s);
    }

    private char getCharacter(int position) {
        return alphabet.get(position).charAt(0);
    }
}
