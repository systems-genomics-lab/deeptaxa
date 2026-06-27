MODULES := index prediction training architecture analysis

.PHONY: preview render clean list $(MODULES) validation

preview:
	quarto preview

render:
	quarto render

$(MODULES):
	quarto render $@.qmd

validation:
	quarto render validation.qmd
	quarto render validation/v3-v4.ipynb
	quarto render validation/full-length.ipynb

list:
	@echo "Available modules: $(MODULES) validation"
	@echo "Usage: make <module>  (e.g., make analysis)"

clean:
	rm -rf _site .quarto
