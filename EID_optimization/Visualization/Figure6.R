library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)

df <- read_csv("C:/Users/wangx/Desktop/network/omission_evaluation_final_summary.csv")
df <- df %>%
  mutate(
    network_name = factor(
      network_name,
      levels = c(
        "Modular", "Scale-free", "University",
        "High school", "Facebook", "Wildbird"
      )
    ),
    type = factor(type, levels = c("edges", "nodes")),
    removal_pct = as.numeric(removal_pct),
    num_sentinels = factor(num_sentinels),
    strategy = factor(strategy)
  )
df_plot <- df %>% filter(num_sentinels == "3")

strategy_colors <- c(
  "RFSM"    = "#BC3C29FF",
  "Greedy" = "#0072B5FF",
  "GA"     = "#E18727FF",
  "Global" = "#20854EFF",
  "Random" = "#FFDC91FF",
  "Modular"= "#7876B1FF"
)

strategy_markers <- c(
  "RFSM"    = 16,
  "GA"      = 16,
  "Global"  = 16,
  "Greedy"  = 16,
  "Modular" = 16,
  "Random"  = 16
)

legend_order <- c(
  "GA",
  "Greedy",
  "RFSM",
  "Modular",
  "Global",
  "Random"
)
base_plot <- function(data, row_title, show_x = FALSE) {
  
  ggplot(
    data,
    aes(
      x = removal_pct,
      y = surveillance_performance_mean,
      color = strategy,
      fill  = strategy,
      shape = strategy,
      group = strategy
    )
  ) +
    geom_ribbon(
      aes(
        ymin = surveillance_performance_mean - surveillance_performance_std,
        ymax = surveillance_performance_mean + surveillance_performance_std
      ),
      alpha = 0.18,
      color = NA
    ) +
    geom_line(linewidth = 0.5) +
    geom_point(size = 1) +
    
    facet_wrap(
      ~ network_name,
      ncol = 3,
      scales = "free_y",
      axes = "all",
      labeller = labeller(
        network_name = c(
          "Modular"     = "i. Modular",
          "Scale-free"  = "ii. Scale-free",
          "University"  = "iii. University",
          "High school" = "iv. High school",
          "Facebook"    = "v. Facebook",
          "Wildbird"    = "vi. Wildbird"
        )
      )
    ) +
    
    scale_color_manual(values = strategy_colors, breaks = legend_order) +
    scale_fill_manual(values = strategy_colors, breaks = legend_order) +
    scale_shape_manual(values = strategy_markers, breaks = legend_order) +
    
    scale_x_continuous(
      breaks = c(0, 20, 40, 60, 80),
      limits = c(0, 90)
    ) +
    
    labs(
      x = if (show_x) "Removal percentage (%)" else NULL,
      y = "Surveillance performance\n(% cases prevented)",
      color = "Strategy",
      fill  = "Strategy",
      shape = "Strategy"
    ) +
    
    ggtitle(row_title) +
    
    guides(
      color = guide_legend(nrow = 1),
      fill  = guide_legend(nrow = 1),
      shape = guide_legend(nrow = 1)
    ) +
    
    theme_bw(base_size = 15) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(colour = "black"),
      
      axis.text.x  = if (show_x) element_text() else element_blank(),
      axis.ticks.x = element_line(),
      
      strip.background = element_rect(fill = NA, color = NA),
      strip.text = element_text(hjust = 0, size = 15),
      plot.title = element_text(face = "bold", hjust = -0.1),
      legend.position = "bottom",
      legend.box = "horizontal"
    )
}

p_edges <- base_plot(
  df_plot %>% filter(type == "edges"),
  "A. Edges removal",
  show_x = TRUE
)

p_nodes <- base_plot(
  df_plot %>% filter(type == "nodes"),
  "B. Nodes removal",
  show_x = TRUE
)

final_plot <-
  p_edges / p_nodes +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

