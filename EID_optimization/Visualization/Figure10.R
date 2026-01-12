library(ggplot2)
library(dplyr)
library(stringr)
library(patchwork)
library(cowplot)

path <- "D:/data_generating/category_importance/"
files <- list.files(path, pattern = "_category_importance.csv$", full.names = TRUE)

data_list <- lapply(files, function(f) {
  fname <- basename(f)
  param <- str_extract(fname, ".*(?=_G[0-9]+_category_importance\\.csv)")
  group <- str_extract(fname, "G[0-9]+")
  df <- read.csv(f, stringsAsFactors = FALSE)
  if (nrow(df) == 0) return(NULL)
  df$param <- param
  df$group <- group
  return(df)
})
data_all <- bind_rows(Filter(Negate(is.null), data_list))

param_order <- c("num_nodes","avg_degree", "heter", 
                 "p", "prob_kurtosis",  "corr","R0")
param_titles <- c(
  "A. Total number of nodes (N)",
  "B. Mean degree (μ)",
  "C. Degree heterogeneity (σ)",
  "D. Within-module connection probability (p)",
  "E. Kurtosis of the prob. distribution",
  "F. Correlation (ρ)", 
  expression(bold(G*". Basic reproduction " * "("* R[0] *")"))
)
data_all$param <- factor(data_all$param, levels = param_order)

category_map <- c(
  "Node topology" = "Node topo.",
  "Global topology" = "Global topo.",
  "Global probability" = "Global prob.",
  "Node probability & topology" = "Node prob.&topo.",
  "Node probability" = "Node prob.",
  "Selection dynamics" = "Selection dynamics"
)
data_all$category <- recode(data_all$category, !!!category_map)

plot_fun <- function(df_param, title_label) {
  if (is.null(df_param) || nrow(df_param) == 0) return(NULL)
  df_param$group <- factor(df_param$group, levels = paste0("G", 1:5))
  df_param$category <- factor(df_param$category, levels = c(
    'Global prob.', 'Global topo.','Node prob.',
    'Node topo.','Node prob.&topo.','Selection dynamics'
  ))
  ggplot(df_param, aes(x = category, y = importance_mean, fill = group)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = importance_mean - importance_std,
                      ymax = importance_mean + importance_std),
                  position = position_dodge(width = 0.8), width = 0.3) +
    scale_fill_brewer(palette = "Blues", na.value = "grey80") +
    labs(title = title_label, x = NULL, y = NULL) +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(),
      plot.title = element_text(hjust = 0, size = 13, face = "bold"),
      legend.title = element_blank()
    )
}

plots <- lapply(seq_along(param_order), function(i) {
  df_p <- data_all %>% filter(param == param_order[i])
  plot_fun(df_p, param_titles[[i]])
})
plots <- Filter(function(x) inherits(x, "ggplot"), plots)


plots_mod <- lapply(seq_along(plots), function(i) {
  p <- plots[[i]]
  if (i <= 5) {
    p <- p + theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2)
    )
  } else if (i %in% c(6,7)) {
    p <- p + theme(
      axis.text.x = element_text(color = "black", size = 12, angle = 90, hjust = 1, vjust = 0),
      legend.position = "none",
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2)
    )
  }
  return(p)
})

plots_mod[[8]] <- ggplot() + theme_void()

left_column <- plot_grid(
  plots_mod[[1]], plots_mod[[3]],
  plots_mod[[5]], plots_mod[[7]],
  ncol = 1, align = "v", rel_heights = c(1, 1, 1, 2)
)

right_column <- plot_grid(
  plots_mod[[2]], plots_mod[[4]],
  plots_mod[[6]], plots_mod[[8]],
  ncol = 1, align = "v", rel_heights = c(1, 1, 2, 1)
)

combined_plot <- plot_grid(left_column, right_column, ncol = 2, rel_widths = c(1, 1))

legend_plot <- plots[[6]] + 
  theme(legend.position = "right",
        legend.text = element_text(size = 12),
        legend.key.size = unit(0.7, "cm")) +
  guides(fill = guide_legend(ncol = 5))
legend_F <- cowplot::get_legend(legend_plot)

final_plot <- ggdraw() +
  draw_plot(combined_plot, x = 0.05, y = 0.05, width = 0.9, height = 0.9) +
  draw_plot(legend_F, x = 0.68, y = 0.1, width = 0.1, height = 0.2) +
  draw_label("Category", x = 0.5, y = 0.02, vjust = 0, size = 13) +
  draw_label("Importance", x = 0.02, y = 0.5, vjust = 1, angle = 90, size = 13) +
  theme(plot.background = element_rect(fill = "white", colour = NA))

