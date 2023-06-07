import java.io.File;

public class Play {
    public static void main(String[] args) {
        String path = System.getProperty("user.dir") + File.separator + "data"
                + File.separator + "q_table_10000000epi_0.5a_0.95g.json";
        QLearningAgent agent = new QLearningAgent(null, 0.5, 0.95, 0.01);

        try {
            agent.setQTable(Loader.load(path.toString()));
        } catch (Exception e) {
            System.out.println("Error saving game: " + e.getMessage());
        }
        Checkers checkers = new Checkers(true, true);

        while (true) {
            checkers.reset();
            while (true) {
                int[][] action = agent.chooseAction(checkers, true, false);
                if (action != null) {
                    boolean keepGoing = checkers.applyMove(action);

                } else {
                    int winner = checkers.getWinner();
                    if (winner == 1) {
                        System.out.println("You win!");
                    } else if (winner == 2) {
                        System.out.println("You lose!");
                    } else {
                        System.out.println("Draw!");
                    }
                    break;
                }
            }

        }

    }
}
