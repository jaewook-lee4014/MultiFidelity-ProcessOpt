```latex
% Requires: \usepackage{booktabs}

% === COFs ===
\begin{table}[htbp]
\centering
\caption{Mean regret at different budgets on \textbf{COFs} benchmark.}
\label{tab:regret_cofs}
\begin{tabular}{lccc}
\toprule
Model & $B=10$ & $B=20$ & $B=30$ \\
\midrule
End-to-End Joint & \textbf{2.9616} & \textbf{0.9429} & \textbf{0.4117} \\
Sequential & 3.3716 & 1.2795 & 0.6376 \\
Pretrain-then-Joint & 3.7487 & 1.6628 & 0.6955 \\
Stop-Gradient Joint & 3.5236 & 1.5532 & 0.7847 \\
Pseudo-Labeling & 3.4117 & 1.4172 & 0.7909 \\
Domain Adaptation (MMD) & 3.0659 & 1.8165 & 0.8045 \\
Curriculum & 3.7421 & 1.5763 & 0.8216 \\
Progressive & 4.1051 & 1.5811 & 0.9651 \\
Soft Parameter Sharing & 3.0408 & 1.8510 & 1.0390 \\
Knowledge Distillation & 3.0117 & 1.8020 & 1.1707 \\
Adapter & 5.2229 & 2.8214 & 1.1758 \\
MFGP & 6.6800 & 6.6800 & 6.5499 \\
\bottomrule
\end{tabular}
\end{table}

% === FreeSolv ===
\begin{table}[htbp]
\centering
\caption{Mean regret at different budgets on \textbf{FreeSolv} benchmark.}
\label{tab:regret_freesolv}
\begin{tabular}{lccccc}
\toprule
Model & $B=10$ & $B=20$ & $B=30$ & $B=40$ & $B=50$ \\
\midrule
End-to-End Joint & 1.1345 & \textbf{0.3090} & \textbf{0.2840} & \textbf{0.2840} & \textbf{0.2840} \\
Curriculum & 1.2765 & 0.4070 & 0.2960 & 0.2860 & 0.2860 \\
Domain Adaptation (MMD) & 1.1970 & 0.3495 & 0.3100 & 0.2880 & 0.2865 \\
Knowledge Distillation & 1.1200 & 0.4215 & 0.3105 & 0.2870 & 0.2870 \\
Stop-Gradient Joint & 1.1835 & 0.4670 & 0.3290 & 0.3175 & 0.2870 \\
Pretrain-then-Joint & 1.5615 & 0.3490 & 0.2900 & 0.2875 & 0.2875 \\
Progressive & 1.5380 & 0.3465 & 0.3005 & 0.2880 & 0.2875 \\
Pseudo-Labeling & 1.2160 & 0.3720 & 0.3230 & 0.2960 & 0.2935 \\
Sequential & \textbf{0.8320} & 0.3220 & 0.3105 & 0.2970 & 0.2950 \\
Soft Parameter Sharing & 1.4275 & 0.3640 & 0.3100 & 0.2980 & 0.2980 \\
Adapter & 4.6260 & 0.7740 & 0.3565 & 0.3370 & 0.3100 \\
MFGP & 2.4330 & 0.8695 & 0.8000 & 0.8000 & 0.8000 \\
\bottomrule
\end{tabular}
\end{table}

% === Polarizability ===
\begin{table}[htbp]
\centering
\caption{Mean regret at different budgets on \textbf{Polarizability} benchmark.}
\label{tab:regret_polarizability}
\begin{tabular}{lccc}
\toprule
Model & $B=10$ & $B=20$ & $B=30$ \\
\midrule
Curriculum & 0.0739 & 0.0650 & \textbf{0.0580} \\
End-to-End Joint & 0.0722 & 0.0675 & \textbf{0.0580} \\
Knowledge Distillation & \textbf{0.0720} & 0.0675 & \textbf{0.0580} \\
Pretrain-then-Joint & \textbf{0.0720} & 0.0675 & \textbf{0.0580} \\
Progressive & \textbf{0.0720} & 0.0643 & \textbf{0.0580} \\
Pseudo-Labeling & \textbf{0.0720} & 0.0675 & \textbf{0.0580} \\
Sequential & 0.0722 & \textbf{0.0612} & \textbf{0.0580} \\
Soft Parameter Sharing & \textbf{0.0720} & 0.0663 & \textbf{0.0580} \\
Stop-Gradient Joint & 0.0752 & 0.0675 & 0.0601 \\
Domain Adaptation (MMD) & \textbf{0.0720} & 0.0643 & 0.0612 \\
Adapter & 0.0919 & 0.0624 & 0.0624 \\
MFGP & 0.2185 & 0.1833 & 0.1702 \\
\bottomrule
\end{tabular}
\end{table}
```
