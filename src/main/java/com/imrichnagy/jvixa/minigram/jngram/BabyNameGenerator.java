package com.imrichnagy.jvixa.minigram.jngram;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static com.imrichnagy.jvixa.minigram.jngram.NativeNgram.NAME_ALPHABET;
import static com.imrichnagy.jvixa.minigram.jngram.NativeNgram.getPosition;

public class BabyNameGenerator {

    private static final String FILE_PATH = "/resources/text/baby_names/";

    private static final int SLICE = 3;
    private static final boolean WEIGHTED = false;

    private static final int FROM_YEAR = 1880; // 1880 minimum
    private static final int TO_YEAR = 1970; // 2022 maximum

    private static final boolean MALE = true;
    private static final boolean FEMALE = true;

    private static final int NAME_COUNT = 20;

    public static void main(String[] args) throws IOException {
        List<String> fileNames = new ArrayList<>();
        for (int i = FROM_YEAR; i <= TO_YEAR; i++) {
            fileNames.add(FILE_PATH + "yob" + i + ".txt");
        }
        NativeNgram ngram = loadBigram(fileNames);
        for (int i = 0; i < NAME_COUNT; i++) {
            generateName(ngram);
        }
    }

    private static void generateName(NativeNgram bigram) {
        StringBuilder name = new StringBuilder();
        String current = ":".repeat(SLICE);
        while (!current.endsWith(";")) {
            char next = bigram.getNextCharacter(current);
            name.append(next);
            current = current.substring(1) + next;
        }
        System.out.println(name.substring(0, 1).toUpperCase() + name.substring(1, name.length() - 1));
    }

    private static NativeNgram loadBigram(List<String> fileNames) throws IOException {
        List<Character> alphabet = NAME_ALPHABET;
        int size = alphabet.size();
        int[][] frequencies = new int[(int) Math.pow(size, SLICE)][size];

        for (String fileName : fileNames) {
            try (Stream<String> stream = Files.lines(Paths.get(System.getProperty("user.dir") + fileName))) {
                stream.forEach((line) -> {
                    String[] data = line.split(",");
                    String sex = data[1];
                    if (sex.equals("M") && !MALE) {
                        return;
                    }
                    if (sex.equals("F") && !FEMALE) {
                        return;
                    }

                    int count = Integer.parseInt(data[2]);
                    String name = ":".repeat(SLICE) + data[0].toLowerCase() + ";";

                    for (int i = 0; i < name.length() - SLICE; i++) {
                        frequencies
                                [getPosition(name.substring(i, i + SLICE), alphabet)]
                                [alphabet.indexOf(name.charAt(i + SLICE))]
                                += WEIGHTED ? count : 1;
                    }
                });
            }
        }

        return new NativeNgram(alphabet, frequencies);
    }
}
