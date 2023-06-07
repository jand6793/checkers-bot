import java.lang.Math;
import java.io.File;

public class Main {
    public static void main(String[] args) {
        int numEpisodes = 20000000;
        double alpha = 0.5;
        double gamma = 0.95;
        double startEpsilon = 1.0;
        double endEpsilon = 0.01;

        Checkers checkers = new Checkers(true, true);
        QLearningAgent agent = new QLearningAgent(checkers, alpha, gamma, startEpsilon);
        String path = System.getProperty("user.dir") + File.separator + File.separator + "data"
                + File.separator + "q_table_" + numEpisodes + "epi_" + alpha + "a_" + gamma + "g" + ".json";

        for (int episode = 0; episode < numEpisodes; episode++) {
            double epsilon = startEpsilon -
                    ((startEpsilon - endEpsilon) * ((double) episode / numEpisodes));
            agent.setEpsilon(epsilon);

            int counter = 0;
            while (true) {
                counter++;
                int status = agent.learn();
                if (counter > 200 || status == 0) {
                    checkers.reset();
                    break;
                }
            }

            if (episode % 10000 == 9999) {
                System.out.println(
                        "Episode " + (episode + 1) + "/" + numEpisodes + " "
                                + Math.round((double) episode / numEpisodes * 100) + "%");
            }
        }

        System.out.println("Training completed.");
        try {
            Loader.save(path.toString(), agent.qTable());
        } catch (Exception e) {
            System.out.println("Error saving game: " + e.getMessage());
        }
    }
}
