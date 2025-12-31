library(igraph)
library(ggraph)
library(ggplot2)
library(RColorBrewer)
library(cowplot)
library(scales)

pal <- brewer.pal(9, "Blues")
col_fun <- col_numeric(palette = pal, domain = c(0, 1))
fixed_col <- col_fun(0.1)  # homogenous

plot_network <- function(graph_file, homogenous = FALSE, title = "", with_legend = FALSE, sentinel_ids = NULL, auto_top_degree = FALSE) {
  g <- read_graph(graph_file, format = "gml")
  
  V(g)$degree <- degree(g)
  V(g)$degree_centrality <- degree(g) / (vcount(g) - 1)
  
  if (homogenous) {
    V(g)$probability <- 0.1
  } else {
    V(g)$probability <- as.numeric(V(g)$probability)
  }
  
  set.seed(42)
  layout_fr <- layout_with_fr(g)
  
  # sentinel nodes
  V(g)$sentinel <- FALSE
  if (auto_top_degree) {
    top_nodes <- order(V(g)$degree, decreasing = TRUE)[1:6]
    V(g)$sentinel[top_nodes] <- TRUE
  } else {
    if (is.null(sentinel_ids)) stop("Please input sentinel_ids")
    sentinel_idx <- match(sentinel_ids, V(g)$id)  # GML id -> igraph
    V(g)$sentinel[sentinel_idx] <- TRUE
  }
  
  # network plot
  p <- ggraph(g, layout = layout_fr) +
    geom_edge_fan(color = "black", width = 0.3)
  
  if (homogenous) {
    p <- p + geom_node_point(
      aes(size = degree_centrality,
          color = factor(sentinel, levels = c(TRUE, FALSE))),
      shape = 21,
      fill = fixed_col,
      stroke = 0.8
    )
  } else {
    p <- p + geom_node_point(
      aes(size = degree_centrality,
          fill = probability,
          color = factor(sentinel, levels = c(TRUE, FALSE))),
      shape = 21,
      stroke = 0.8
    )
  }
  
  # legend setup
  p <- p +
    scale_fill_gradientn(
      colours = pal,
      name = "Emergence probability"
    ) +
    scale_size(
      range = c(3.5, 8),
      name = "Degree centrality"
    ) +
    scale_color_manual(
      name = "Node type",
      values = c("TRUE" = "red", "FALSE" = "grey30"),
      labels = c("TRUE" = "Sentinel nodes\nselected by GA", "FALSE" = "Other nodes"),
      guide = guide_legend(
        override.aes = list(size = 6),
        order = 3
      )
    ) +
    guides(
      fill = guide_colorbar(order = 1, barwidth = 0.6, barheight = 6),
      size = guide_legend(order = 2)
    ) +
    theme_void() +
    ggtitle(title) +
    theme(
      plot.title = element_text(face = "bold", size = 17, hjust = 0),
      legend.position = if (with_legend) "right" else "none",
      legend.direction = "vertical"
    )
  
  return(p)
}

pA <- plot_network("C:\\Users\\wangx\\Desktop\\network\\modified_network.gml",
                   homogenous = TRUE,
                   title = "A. Modular, homogenous probability",
                   auto_top_degree = TRUE)

pB <- plot_network("C:\\Users\\wangx\\Desktop\\network\\modified_network.gml",
                   homogenous = FALSE,
                   title = "B. Modular, heterogeneous probability",
                   with_legend = TRUE,
                   sentinel_ids = c(17, 62, 63, 39, 58, 66))

pC <- plot_network("C:\\Users\\wangx\\Desktop\\network\\BA_network-modified.gml",
                   homogenous = TRUE,
                   title = "C. Scale-free, homogenous probability",
                   auto_top_degree = TRUE)

pD <- plot_network("C:\\Users\\wangx\\Desktop\\network\\BA_network-modified.gml",
                   homogenous = FALSE,
                   title = "D. Scale-free, heterogeneous probability",
                   sentinel_ids = c(74, 34, 7, 11, 2, 89))

legend_b <- get_legend(pB + theme(legend.position = "right",
                                  legend.text = element_text(size = 12),
                                  legend.title = element_text(size = 13)
                                  )
                       )

pA <- pA + theme(legend.position = "none")
pB <- pB + theme(legend.position = "none")
pC <- pC + theme(legend.position = "none")
pD <- pD + theme(legend.position = "none")

plot_grid_main <- plot_grid(
  plot_grid(pA, pB, ncol = 2),
  plot_grid(pC, pD, ncol = 2),
  ncol = 1,
  rel_heights = c(1, 1)
)

final_plot <- plot_grid(
  plot_grid_main, legend_b, 
  ncol = 2, 
  rel_widths = c(1, 0.15)
)

print(final_plot)
