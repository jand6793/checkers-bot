import java.io.File;
import java.util.*;

public class Table {
    public static void main(String[] args) {
        String path = System.getProperty("user.dir") + File.separator + "data"
                + File.separator + "q_table_3000000epi_0.5a_0.95g.json";
        Map<String, Double> qTable;

        try {
            qTable = Loader.load(path.toString());
        } catch (Exception e) {
            System.out.println("Error saving game: " + e.getMessage());
            return;
        }

        List<Double> qValues = new ArrayList<Double>(qTable.values());
        List<Double> significantlyLearnedValues = new ArrayList<Double>();
        for (Double qValue : qValues) {
            if (qValue > 0.5 || qValue < -0.5) {
                significantlyLearnedValues.add(qValue);
            }
        }
        List<Double> sortedSignificantlyLearnedValues = new ArrayList<Double>(significantlyLearnedValues);
        Collections.sort(sortedSignificantlyLearnedValues);
        int test = 0;
    }
}
