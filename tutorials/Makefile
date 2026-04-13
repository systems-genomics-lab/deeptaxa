MODULES := index prediction training architecture analysis

.PHONY: preview render clean list $(MODULES)

preview:
	quarto preview

render:
	quarto render

$(MODULES):
	quarto render $@.qmd

list:
	@echo "Available modules: $(MODULES)"
	@echo "Usage: make <module>  (e.g., make analysis)"

clean:
	rm -rf _site .quarto
