### Graph-Based SpMM Design (Thresholded)

**Objective**: Compute $C = A \times B$ with on-the-fly thresholding to maintain sparsity and avoid unnecessary communication/computation.
**Constraint**: 1D distribution (Block-Rows), Memory Limited.

#### 1. Data Structures
-   **BlockSpMat**: Existing structure.
    -   `row_ptr`, `col_ind`, `val`, `blk_ptr`.
    -   `graph`: Pointer to `DistGraph`.
-   **DistGraph**:
    -   `owned_global_indices`: List of global rows owned by this rank.
    -   `global_to_local`: Map global -> local.
    -   `comm`: MPI communicator.

#### 2. Algorithm Steps

**Phase 1: Preprocessing (Norm Computation)**
-   Compute Frobenius norms of all blocks in $A$ and $B$.
-   Store as lightweight `vector<double> block_norms`.
-   *Optimization*: If norms are already cached/maintained, skip this.

**Phase 2: Symbolic Filter (Metadata Exchange)**
-   **Goal**: Determine which blocks of $C$ will be significant ($> \epsilon$) and which ghost blocks of $B$ are needed.
-   **Step 2a: Identify Needed Ghost Rows of B**
    -   $A$ needs rows of $B$ corresponding to its column indices ($A_{ik} \neq 0 \implies$ need row $k$ of $B$).
    -   Identify unique column indices in $A$ that are *not* locally owned. These are the "Ghost Rows of B" we need.
-   **Step 2b: Request Metadata**
    -   Send requests to owners of these ghost rows.
    -   Request: "Send me the *structure* (col indices) and *norms* of row $k$".
    -   *Note*: Do NOT send actual data yet.
-   **Step 2c: Receive Metadata & Predict Structure**
    -   Receive `(col_idx, norm)` pairs for ghost rows.
    -   **Symbolic Multiplication with Thresholding**:
        -   For each local row $i$ of $A$:
            -   Initialize map `C_row_estimates[j] = 0.0`.
            -   For each $k$ where $A_{ik} \neq 0$:
                -   $NormA = A_{ik}.norm$
                -   For each $j$ in row $k$ of $B$ (using received metadata):
                    -   $NormB = B_{kj}.norm$
                    -   `C_row_estimates[j] += NormA * NormB`
    -   **Filter**:
        -   If `C_row_estimates[j] > threshold`:
            -   Mark $C_{ij}$ as a non-zero.
            -   Mark $B_{kj}$ as "Required Data".
        -   Else:
            -   Ignore.

**Phase 3: Data Exchange (Fetch Required Blocks)**
-   **Goal**: Fetch only the $B_{kj}$ blocks that contribute to significant $C_{ij}$.
-   **Step 3a: Request Data**
    -   Send list of "Required Blocks" (GlobalRow $k$, GlobalCol $j$) to owners.
    -   *Optimization*: If we need *most* of a row, fetch the whole row. If sparse, fetch individual blocks.
-   **Step 3b: Receive Data**
    -   Store received blocks in a temporary `GhostBlockBuffer`.

**Phase 4: Numerical Multiplication**
-   **Goal**: Compute values for the predicted structure.
-   **Step 4a: Execution**
    -   Iterate over local rows $i$ of $A$.
    -   For each $k$ where $A_{ik} \neq 0$:
        -   Get pointer to $A_{ik}$.
        -   For each $j$ in row $k$ of $B$ (if $B_{kj}$ was fetched):
            -   Get pointer to $B_{kj}$ (local or ghost buffer).
            -   Accumulate: $C_{ij} += A_{ik} \times B_{kj}$ (Dense GEMM).
-   **Step 4b: Final Thresholding**
    -   After accumulation, check exact norm of $C_{ij}$.
    -   If $< \epsilon$, drop it (hard zero).

#### 3. Implementation Plan
1.  **Helper**: `compute_block_norms(matrix)` -> `vector<double>`.
2.  **Communication**: Implement `exchange_row_metadata` (indices + norms).
3.  **Symbolic**: Implement `predict_structure_and_filter(A, B_metadata, threshold)`.
4.  **Communication**: Implement `fetch_specific_blocks`.
5.  **Numerical**: Implement `multiply_filtered`.

#### 4. Complexity Analysis
-   **Communication**:
    -   Metadata: Small (integers + doubles).
    -   Data: Only "significant" blocks. Much less than full 2.5D replication.
-   **Computation**:
    -   Symbolic: Integer ops + scalar muls. Fast.
    -   Numerical: Dense GEMM on reduced set. High efficiency.
-   **Memory**:
    -   Stores $A$ (local), $C$ (local), and *subset* of $B$ (ghosts). Fits in 128GB.
