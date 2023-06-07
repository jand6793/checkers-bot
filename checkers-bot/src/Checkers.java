import java.util.*;

public class Checkers {
    private boolean switchPlayer;
    private int[][] board;
    private int currentPlayer;
    private Map<String, Integer> visited = new HashMap<>();
    private String stateString;
    private String actionString;
    private String fullStateString;
    private String[] fullValidStateStrings;
    private String[] prevFullValidStateStrings;
    private boolean prevWasJump;
    private List<int[][]> validMoves;
    private List<int[][]> prevValidMoves;
    private int numP1Pieces;
    private int numP2Pieces;

    public Checkers(boolean switchPlayer, boolean initialize) {
        this.switchPlayer = switchPlayer;
        if (initialize) {
            reset();
            calcState();
        }
    }

    private void calcState() {
        stateString = stateToString();
        fullStateString = calcFullStateString();
        fullValidStateStrings = calcFullStateStrings();
        prevFullValidStateStrings = calcPrevFullStateStrings();
        validMoves = calcValidMoves(currentPlayer);
        prevValidMoves = calcValidMoves(oppPlayer(currentPlayer));
    }

    public boolean inBounds(int x, int y) {
        return 0 <= x && x < 8 && 0 <= y && y < 8;
    }

    public List<int[][]> calcJumps(int x, int y, List<int[]> visited) {
        if (visited == null) {
            visited = new ArrayList<>();
        }
        visited.add(new int[] { x, y });

        List<int[][]> jumps = new ArrayList<>();
        for (int[] direction : getDirections(board[y][x])) {
            int dx = direction[0], dy = direction[1];
            int newX = x + dx, newY = y + dy;
            int jumpX = newX + dx, jumpY = newY + dy;

            if (inBounds(newX, newY) && inBounds(jumpX, jumpY)
                    && board[newY][newX] == oppPlayer(currentPlayer) && board[jumpY][jumpX] == 0
                    && !visited.contains(new int[] { jumpX, jumpY })) {

                visited.add(new int[] { jumpX, jumpY });
                jumps.add(new int[][] { { x, y }, { jumpX, jumpY } });
                jumps.addAll(calcJumps(jumpX, jumpY, visited));
            }
        }

        return jumps;
    }

    // Get a list of valid moves (jumps or regular moves) for the current player
    private List<int[][]> calcValidMoves(int player) {
        List<int[][]> moves = new ArrayList<>();
        List<int[][]> jumps = new ArrayList<>();

        // Find positions of all pieces for the current player
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                // if (board[y][x] * player > 0) {
                if ((board[y][x] == 1 || board[y][x] == 2) && player == 1
                        || (board[y][x] == 3 || board[y][x] == 4) && player == 2) {
                    for (int[] direction : getDirections(board[y][x])) {
                        int dx = direction[0], dy = direction[1];
                        int newX = x + dx, newY = y + dy;

                        if (inBounds(newX, newY) && board[newY][newX] == 0) {
                            moves.add(new int[][] { { x, y }, { newX, newY } });
                        }

                        jumps.addAll(calcJumps(x, y, null));
                    }
                }
            }
        }

        List<int[][]> validJumps = filterActionOccurances(jumps, 3);

        if (!validJumps.isEmpty()) {
            return validJumps;
        } else {
            return filterActionOccurances(moves, 3);
        }
    }

    public int[][] getDirections(int piece) {
        // if (Math.abs(piece) == 2) {
        if (piece == 2 || piece == 4) {
            return new int[][] { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
        } else if (piece == 1) {
            return new int[][] { { -1, 1 }, { 1, 1 } };
        } else {
            return new int[][] { { -1, -1 }, { 1, -1 } };
        }
    }

    private List<int[][]> filterActionOccurances(List<int[][]> validMoves, int maxCount) {
        List<int[][]> filteredMoves = new ArrayList<>();
        for (int[][] move : validMoves) {
            String action = actionToString(move);

            if (getActionOccurences(action) < maxCount) {
                filteredMoves.add(move);
            }
        }
        return filteredMoves;
    }

    private String stateToString() {
        int[] flattenedBoard = flattenBoard();
        StringBuilder stateBuilder = new StringBuilder(); // Continue this
        for (int i = 0; i < 8; i++) {
            for (int j = (i + 1) % 2; j < 8; j += 2) {
                stateBuilder.append(flattenedBoard[j + i * 8]);
            }
        }
        for (int piece : flattenedBoard) {
            if (piece != 0 && piece != 1 && piece != 2 && piece != 3 && piece != 4) {
                stateBuilder.append(piece);
            }
        }
        return stateBuilder.toString();
    }

    /**
     * @return true if the game is not over, false otherwise
     */
    public boolean applyMove(int[][] action) {
        int startX = action[0][0], startY = action[0][1];
        int endX = action[1][0], endY = action[1][1];
        board[endY][endX] = board[startY][startX];
        board[startY][startX] = 0;

        // Check if a jump was made, and remove the captured piece
        prevWasJump = isJump(action);
        if (prevWasJump) {
            int capturedX = (startX + endX) / 2;
            int capturedY = (startY + endY) / 2;
            board[capturedY][capturedX] = 0;
        }

        // Check if a piece became a king
        if ((endY == 0 && board[endY][endX] == 3)) {
            board[endY][endX] = 4;
        } else if ((endY == 7 && board[endY][endX] == 1)) {
            board[endY][endX] = 2;
        }

        if (prevWasJump) {
            if (currentPlayer == 1) {
                --numP2Pieces;
            } else {
                --numP1Pieces;
            }
        } else if (switchPlayer) {
            currentPlayer = oppPlayer(currentPlayer);
        }

        stateString = stateToString();
        actionString = actionToString(action);
        fullStateString = calcFullStateString();
        validMoves = calcValidMoves(currentPlayer);
        prevValidMoves = calcValidMoves(oppPlayer(currentPlayer));
        fullValidStateStrings = calcFullStateStrings();
        prevFullValidStateStrings = calcPrevFullStateStrings();
        int actionOccurances = getActionOccurences(actionString);
        visited.put(fullStateString, ++actionOccurances);
        return prevWasJump && !validMoves.isEmpty();
    }

    private String calcFullStateString() {
        return stateString + "" + actionString;
    }

    public boolean isJump(int[][] action) {
        return Math.abs(action[0][0] - action[1][0]) == 2;
    }

    private int getActionOccurences(String action) {
        return visited.getOrDefault(stateString + "" + action, 0);
    }

    public void reset() {
        board = new int[8][8];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                if ((y % 2 == 0 && x % 2 != 0) || (y % 2 != 0 && x % 2 == 0)) {
                    if (y < 3)
                        board[y][x] = 1;
                    if (y > 4)
                        board[y][x] = 3;
                }
            }
        }
        currentPlayer = 1;
        visited.clear();
        stateString = stateToString();
        actionString = null;
        fullStateString = null;
        fullValidStateStrings = null;
        prevFullValidStateStrings = null;
        prevWasJump = false;
        validMoves = calcValidMoves(currentPlayer);
        prevValidMoves = calcValidMoves(oppPlayer(currentPlayer));
        numP1Pieces = 12;
        numP2Pieces = 12;
    }

    public String[] calcFullStateStrings() {
        return baseCalcFullStateStrings(validMoves);
    }

    public String[] calcPrevFullStateStrings() {
        return baseCalcFullStateStrings(prevValidMoves);
    }

    private String[] baseCalcFullStateStrings(List<int[][]> moves) {
        List<String> validActionStrings = new ArrayList<>();
        for (int[][] move : moves) {
            validActionStrings.add(actionToString(move));
        }
        return validActionStrings.toArray(new String[0]);
    }

    public static String actionToString(int[][] action) {
        return action[0][0] + "" + action[0][1] + action[1][0] + "" + action[1][1];
    }

    public int[][] copyBoard() {
        return Arrays.stream(board).map(int[]::clone).toArray(int[][]::new);
    }

    public int getCurrentPlayer() {
        return currentPlayer;
    }

    public void setCurrentPlayer(int currentPlayer) {
        this.currentPlayer = currentPlayer;
    }

    public int[][] getBoard() {
        return board;
    }

    public static int oppPlayer(int player) {
        return player == 1 ? 2 : 1;
    }

    public int[] flattenBoard() {
        int[] flatBoard = new int[8 * 8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                flatBoard[i * 8 + j] = board[i][j];
            }
        }
        return flatBoard;
    }

    public String getStateString() {
        return stateString;
    }

    public boolean getPrevWasJump() {
        return prevWasJump;
    }

    public List<int[][]> getValidMoves() {
        return validMoves;
    }

    public List<int[][]> getPrevValidMoves() {
        return prevValidMoves;
    }

    public String getActionString() {
        return actionString;
    }

    public String getFullStateString() {
        return fullStateString;
    }

    public String[] getFullValidStateStrings() {
        return fullValidStateStrings;
    }

    public String[] getPrevFullValidStateStrings() {
        return prevFullValidStateStrings;
    }

    public int getWinner() {
        if (numP1Pieces == 0) {
            return 2;
        } else if (numP2Pieces == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    public int getNumP1Pieces() {
        return numP1Pieces;
    }

    public int getNumP2Pieces() {
        return numP2Pieces;
    }

    public Checkers clone(boolean initialize) {
        Checkers clone = new Checkers(switchPlayer, false);
        clone.board = copyBoard();
        clone.currentPlayer = currentPlayer;
        clone.visited = new HashMap<>(visited);
        clone.stateString = stateString;
        clone.actionString = actionString;
        clone.fullStateString = fullStateString;
        clone.fullValidStateStrings = fullValidStateStrings;
        clone.prevFullValidStateStrings = prevFullValidStateStrings;
        clone.prevWasJump = prevWasJump;
        clone.validMoves = new ArrayList<>(validMoves);
        clone.prevValidMoves = new ArrayList<>(prevValidMoves);
        clone.numP1Pieces = numP1Pieces;
        clone.numP2Pieces = numP2Pieces;
        if (initialize) {
            clone.calcState();
        }
        return clone;
    }
}
