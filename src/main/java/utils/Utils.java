package utils;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Utils {
    public static void save(String fileName, ArrayList<Double> list) throws IOException {
        FileWriter writer = null;
        try {
            writer = new FileWriter(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (Double aDouble : list) {
            assert writer != null;
            writer.append(String.valueOf(aDouble));
            writer.append("\n");
        }
        assert writer != null;
        writer.close();
    }
}
