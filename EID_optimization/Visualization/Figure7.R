library(ggplot2)
library(ggsci)
library(patchwork)
library(dplyr)
library(tidyverse)

feature_importance <- read_csv("D:/data_generating/importance_0.3/feature_importance.csv",
                               show_col_types = FALSE) %>% 
  select(feature, category, importance = permutation_importance, error = std)

category_importance <- read_csv("D:/data_generating/importance_0.3/category_importance.csv",
                                show_col_types = FALSE) %>% 
  select(category, importance = importance_mean, std = importance_std)

feature_importance$feature <- factor(
  feature_importance$feature,
  levels = feature_importance$feature[order(feature_importance$importance, decreasing = FALSE)]
)

category_importance$category <- factor(
  category_importance$category,
  levels = category_importance$category[order(category_importance$importance, decreasing = FALSE)]
)


npg_colors <- c(
  "Global prob." = "#E64B35FF",
  "Global topo." = "#4DBBD5FF", 
  "Node prob." = "#00A087FF",
  "Node topo." = "#3C5488FF",
  "Node prob. & topo." = "#F39B7FFF",
  "Selection dynamics" = "#8491B4FF"
)

#npg_colors <- pal_npg()(length(unique(feature_importance$category)))
#names(npg_colors) <- unique(feature_importance$category)

p1 <- ggplot(feature_importance, aes(x = feature, y = importance, fill = category)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = importance - error, ymax = importance + error), width = 0.3) +
  coord_flip() +
  scale_fill_manual(values = npg_colors) +
  labs(title = "A. Feature-level Importance",
       x = NULL, y = NULL, fill = "Category") +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = "black"),
    plot.title = element_text(hjust = 0, face = "bold"),
    legend.position = "none",
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(2, "pt")
  )

p2 <- ggplot(category_importance, aes(x = category, y = importance, fill = category)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = importance - std, ymax = importance + std), width = 0.3) +
  coord_flip() +
  scale_fill_manual(values = npg_colors) +
  labs(title = "B. Category-level Importance",
       x = NULL, y = "Importance", fill = "Category") +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = "black"),
    plot.title = element_text(hjust = 0, face = "bold"),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(2, "pt"),
    legend.position = "none"
  )

p_combined <- p1 / p2 + plot_layout(heights = c(2, 1))
p_combined
