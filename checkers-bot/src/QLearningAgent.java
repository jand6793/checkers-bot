import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class QLearningAgent {
    private Checkers checkers;
    private double alpha;
    private double gamma;
    private double epsilon;
    private Map<String, Double> qTable = new HashMap<>();

    public QLearningAgent(Checkers checkers, double alpha, double gamma, double epsilon) {
        this.checkers = checkers;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }

    public double getQValue(String state) {
        return qTable.getOrDefault(state, 0.0);
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    /**
     * @return 0 if there are no available moves, 1 if the taken action is not a
     *         jump, 2 if you need to jump again
     */
    public int applyAction() {
        int[][] action = chooseAction(checkers, false, false);
        if (action == null) {
            return 0;
        } else {
            boolean continueJumping = checkers.applyMove(action);
            return continueJumping ? 2 : 1;
        }
    }

    public int[][] chooseAction(Checkers checkers, boolean greedy, boolean nextPlayer) {
        List<int[][]> validMoves = nextPlayer ? checkers.getPrevValidMoves()
                : checkers.getValidMoves();
        if (validMoves.isEmpty()) {
            return null;
        } else if (!greedy && ThreadLocalRandom.current().nextDouble() < epsilon) {
            int randomIndex = ThreadLocalRandom.current().nextInt(validMoves.size());
            return validMoves.get(randomIndex);
        } else {
            double[] qValues = new double[validMoves.size()];
            for (int i = 0; i < validMoves.size(); i++) {
                qValues[i] = getQValue(
                        checkers.getStateString() + Checkers.actionToString(validMoves.get(i)));
            }
            return validMoves.get(argmax(qValues));
        }
    }

    public int calculateReward(boolean continueJumping) {
        if (checkers.getWinner() != 0) {
            return 10;
        } else {
            return (checkers.getPrevWasJump() ? 1 : 0) + calcOpponentReward(continueJumping);
        }
    }

    private int calcOpponentReward(boolean continueJumping) {
        Checkers checkersCopy = checkers.clone(false);
        if (continueJumping) {
            checkersCopy.setCurrentPlayer(Checkers.oppPlayer(checkersCopy.getCurrentPlayer()));
        }
        int reward = 0;
        while (true) {
            int[][] bestOpponentAction = bestOpponentAction(checkersCopy, continueJumping);
            if (bestOpponentAction == null) {
                if (checkersCopy.getWinner() != 0) {
                    return -10;
                } else {
                    return reward;
                }
            }
            boolean actionIsJump = checkersCopy.isJump(bestOpponentAction);
            if (actionIsJump) {
                reward--;
            } else {
                return reward;
            }
            checkersCopy.applyMove(bestOpponentAction);
        }
    }

    /**
     * @return 0 if the game is over, 1 if the game will continue and 2 if you need
     *         to jump again
     */
    public int learn() {
        int moveStatus = applyAction();
        int reward = calculateReward(moveStatus == 2);
        double currentQValue = getQValue(checkers.getFullStateString());
        double maxNextQValue = getMaxPossibleQValue(checkers, moveStatus != 2);
        double newQValue = currentQValue + alpha * (reward + gamma * maxNextQValue - currentQValue);
        if (newQValue != 0) {
            qTable.put(checkers.getFullStateString(), newQValue);
        }
        if (moveStatus == 0 || checkers.getWinner() != 0) {
            return 0;
        } else {
            return moveStatus;
        }
    }

    private double getMaxPossibleQValue(Checkers checkers, boolean next) {
        double max = 0;
        String[] fullStateStrings = next ? checkers.getPrevFullValidStateStrings()
                : checkers.getFullValidStateStrings();
        for (String fullStateString : fullStateStrings) {
            double qValue = getQValue(fullStateString);
            if (qValue > max) {
                max = qValue;
            }
        }
        return max;
    }

    private int[][] bestOpponentAction(Checkers checkers, boolean next) {
        String[] fullStateStrings = next ? checkers.getPrevFullValidStateStrings()
                : checkers.getFullValidStateStrings();
        List<int[][]> validMoves = next ? checkers.getPrevValidMoves()
                : checkers.getValidMoves();
        double max = 0;
        int[][] bestOpponentAction = null;
        for (int i = 0; i < fullStateStrings.length; i++) { // fullStateStrings is incorrect
            double qValue = getQValue(fullStateStrings[i]);
            if (qValue > max) {
                max = qValue;
                bestOpponentAction = validMoves.get(i);
            }
        }
        if (bestOpponentAction == null) {
            if (fullStateStrings.length == 0) {
                return null;
            }
            int randomIndex = ThreadLocalRandom.current().nextInt(fullStateStrings.length);
            return validMoves.get(randomIndex);
        }
        return bestOpponentAction;
    }

    public int argmax(double[] array) {
        double max = array[0];
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }

    public Map<String, Double> qTable() {
        return qTable;
    }

    public void setQTable(Map<String, Double> qTable) {
        this.qTable = qTable;
    }
}
