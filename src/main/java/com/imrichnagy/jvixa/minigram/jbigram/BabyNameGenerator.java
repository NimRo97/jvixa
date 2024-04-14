package com.imrichnagy.jvixa.minigram.jbigram;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static com.imrichnagy.jvixa.minigram.jbigram.NativeNgram.NAME_ALPHABET;

public class BabyNameGenerator {

    private static final String FILE_PATH = "/resources/text/baby_names/";

    private static final boolean WEIGHTED = true;

    private static final int FROM_YEAR = 1880; // 1880 minimum
    private static final int TO_YEAR = 2022; // 2022 maximum

    private static final boolean MALE = true;
    private static final boolean FEMALE = true;

    public static void main(String[] args) throws IOException {
        List<String> fileNames = new ArrayList<>();
        for (int i = FROM_YEAR; i <= TO_YEAR; i++) {
            fileNames.add(FILE_PATH + "yob" + i + ".txt");
        }
        NativeNgram ngram = loadBigram(fileNames);
        for (int i = 0; i < 10; i++) {
            generateName(ngram);
        }
    }

    private static void generateName(NativeNgram bigram) {
        StringBuilder name = new StringBuilder();
        String current = "::";
        while (!current.endsWith(";")) {
            char next = bigram.getNextCharacter(current);
            name.append(next);
            current = current.substring(1) + next;
        }
        System.out.println(name.substring(0, 1).toUpperCase() + name.substring(1, name.length() - 1));
    }

    private static NativeNgram loadBigram(List<String> fileNames) throws IOException {
        List<String> alphabet = new ArrayList<>(NAME_ALPHABET);
        for (String c1 : NAME_ALPHABET) {
            for (String c2 : NAME_ALPHABET) {
                alphabet.add(c1 + c2);
            }
        }
        int size = alphabet.size() * (alphabet.size() + 1);
        long[][] frequencies = new long[size][alphabet.size()];

        for (String fileName : fileNames) {
            try (Stream<String> stream = Files.lines(Paths.get(System.getProperty("user.dir") + fileName))) {
                stream.forEach((line) -> {
                    String[] parts = line.split(",");
                    String sex = parts[1];
                    if (sex.equals("M") && !MALE) {
                        return;
                    }
                    if (sex.equals("F") && !FEMALE) {
                        return;
                    }

                    long count = Integer.parseInt(parts[2]);
                    String name = parts[0].toLowerCase();

                    frequencies[alphabet.indexOf("::")][alphabet.indexOf(name.charAt(0) + "")] += WEIGHTED ? count : 1;
                    frequencies[alphabet.indexOf(":" + name.charAt(0))][alphabet.indexOf(name.charAt(1) + "")]++;
                    for (int i = 0; i < name.length() - 2; i++) {
                        frequencies[alphabet.indexOf("" + name.charAt(i) + name.charAt(i + 1))][alphabet.indexOf(name.charAt(i + 2) + "")] += WEIGHTED ? count : 1;
                    }
                    frequencies[alphabet.indexOf("" + name.charAt(name.length() - 2) + name.charAt(name.length() - 1))][alphabet.indexOf(";")] += WEIGHTED ? count : 1;
                });
            }
        }

        return new NativeNgram(alphabet, frequencies);
    }
}
