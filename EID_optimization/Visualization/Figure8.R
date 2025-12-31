library(ggplot2)
library(tidyr)
library(ggsci)
library(readr)
library(RColorBrewer)

df <- read_csv("D:/data_generating/importance_0.3/rank_category_importance.csv")

df_long <- df %>%
  pivot_longer(
    cols = c(`Node topo.`, `Global topo.`, `Global prob.`,
             `Node prob. & topo.`, `Node prob.`, `Selection dynamics`),
    names_to = "Category",
    values_to = "Importance"
  )

df_long$FillValue <- ifelse(df_long$Importance == 0, NA, df_long$Importance)

plot <- ggplot(df_long, aes(x = factor(Rank), y = Category, fill = FillValue)) +
  geom_tile(color = "white") +
  geom_text(data = ~ subset(.x, Importance != 0), aes(label = sprintf("%.3f", Importance)), color = "white", size = 3) +
  scale_fill_gradientn(
    colours = rev(brewer.pal(11, "RdYlBu")),
    na.value = "grey"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10))
  ) +
  labs(fill = "Importance", x = "Rank", y = "Category") +
  theme(plot.background = element_rect(fill = "white", colour = NA))
print(plot)
ggsave("D:/data_generating/importance_0.3/figure/rank-specific.png", plot, width = 9.5, height = 4, dpi = 320)


df_long <- read_csv("D:/data_generating/importance_0.3/rank_feature_importance.csv")
df_long$FillValue <- ifelse(df_long$Importance == 0, NA, df_long$Importance)

feature_order <- c(
  'neighbor_selected_ratio', 'synergy_score', 'redundancy_score', 'new_coverage_ratio',
  'density', 'avg_degree', 'degree_skewness', 'degree_centrality', 
  'betweenness_centrality', 'eigenvector_centrality', 'local_density',
  'prob_weighted_degree', 'probability', 'prob_mean'
)

df_long$Feature <- factor(df_long$Feature, levels = rev(feature_order), ordered = TRUE)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(scales)

pal <- rev(brewer.pal(11, "RdYlBu"))
df_long <- df_long %>%
  mutate(
    FillColor = ifelse(
      is.na(FillValue),
      "grey",
      pal[cut(FillValue, breaks = length(pal), labels = FALSE, include.lowest = TRUE)]
    )
  )

get_luminance <- function(hex_color) {
  rgb <- col2rgb(hex_color)
  0.2126 * rgb[1,] + 0.7152 * rgb[2,] + 0.0722 * rgb[3,]
}

df_long <- df_long %>%
  mutate(
    TextColor = ifelse(get_luminance(FillColor) > 128, "black", "white")
  )

plot <- ggplot(df_long, aes(x = factor(Rank), y = Feature, fill = FillValue)) +
  geom_tile(color = "white") +
  geom_text(
    data = ~ subset(.x, Importance != 0),
    aes(label = sprintf("%.3f", Importance), color = TextColor),
    size = 3,
    show.legend = FALSE
  ) +
  scale_fill_gradientn(
    colours = pal,
    na.value = "grey"
  ) +
  scale_color_identity() + 
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10))
  ) +
  labs(fill = "Importance", x = "Rank", y = "Feature")

print(plot)
ggsave("D:/data_generating/importance_0.3/figure/rank-specific_feature.png", plot, width = 12, height = 6, dpi = 320)
