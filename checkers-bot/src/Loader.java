import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.util.*;
import java.lang.reflect.Type;

public class Loader {

    public Loader() {}

    public static void save(String fileName, Map<String, Double> map) throws IOException {
        // Create a GsonBuilder, set the pretty printing, and create a Gson object
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        File file = new File(fileName);

        // Create the file if it doesn't exist
        if (!file.exists()) {
            file.createNewFile();
        }

        // Write the QTable to the file as pretty-printed JSON
        try (FileWriter writer = new FileWriter(file)) {
            gson.toJson(map, writer);
        }
    }

    public static Map<String, Double> load(String fileName) throws IOException {
        Gson gson = new Gson();
        File file = new File(fileName);

        if (!file.exists()) {
            throw new IOException("File does not exist");
        }

        // Read the file and convert the JSON back into a Map<String, Double>
        Type mapType = new TypeToken<Map<String, Double>>() {
        }.getType();
        try (FileReader reader = new FileReader(file)) {
            return gson.fromJson(reader, mapType);
        }
    }
}
