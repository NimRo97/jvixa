package com.imrichnagy.jvixa.minigram;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NativeNgram {

    public static final List<Character> NAME_ALPHABET = List.of(
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ';', ':'
    );

    private final List<Character> alphabet;
    private final int[][] frequencies;
    private final Random random = new Random();

    public NativeNgram(List<Character> alphabet, int[][] frequencies) {
        this.alphabet = alphabet;
        this.frequencies = frequencies;
    }

    public Character getNextCharacter(String current) {
        int[] freq = frequencies[getPosition(current)];
        long sum = Arrays.stream(freq).sum();
        if (sum == 0) {
            return ';';
        }
        long r = random.nextLong(sum);
        for (int i = 0; i < freq.length; i++) {
            r -= freq[i];
            if (r <= 0) {
                return getCharacter(i);
            }
        }
        return getCharacter(freq.length - 1);
    }

    public static int getPosition(String s, List<Character> alphabet) {
        int position = 0;
        for (int i = 0; i < s.length(); i++) {
            position = position * alphabet.size() + alphabet.indexOf(s.charAt(i));
        }
        return position;
    }

    private int getPosition(String s) {
        return getPosition(s, alphabet);
    }

    private char getCharacter(int position) {
        return alphabet.get(position);
    }
}
