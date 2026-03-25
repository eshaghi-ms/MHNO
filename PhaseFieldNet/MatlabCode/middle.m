function middleMatrix = middle(inputMatrix, outputSize)
    % Check if the input is a 1D array
    if isvector(inputMatrix)
        % For 1D array, calculate the indices for the middle section
        startIndex = round((length(inputMatrix) - outputSize) / 2) + 1;
        endIndex = startIndex + outputSize - 1;
        % Extract the middle part
        middleMatrix = inputMatrix(startIndex:endIndex);
    elseif ismatrix(inputMatrix)
        % For 2D array, calculate the indices for the middle submatrix
        startIndexRow = round((size(inputMatrix, 1) - outputSize) / 2) + 1;
        endIndexRow = startIndexRow + outputSize - 1;
        startIndexCol = startIndexRow;
        endIndexCol = endIndexRow;
        % Extract the middle submatrix
        middleMatrix = inputMatrix(startIndexRow:endIndexRow, startIndexCol:endIndexCol);
    else
        error('Input must be either a 1D or 2D array.');
    end
end