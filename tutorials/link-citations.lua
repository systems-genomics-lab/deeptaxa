-- Wrap each inline citation in a link to its entry on the central References page.
-- With suppress-bibliography set, citations render as plain text because there is
-- no on-page bibliography to link to; this points them at references.qmd instead,
-- and Quarto resolves the relative path for each page.
function Cite(cite)
  if #cite.citations == 0 then return nil end
  local id = cite.citations[1].id
  return pandoc.Link(cite, 'references.qmd#ref-' .. id)
end
