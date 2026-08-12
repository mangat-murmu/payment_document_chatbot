const byId = (id) => document.querySelector(`#${id}`);
const openModal = (id) => byId(id).showModal();
const documentTypeLabel = (type) => type.replaceAll("_", " ");
const formatBytes = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};
const formatDate = (value) => new Date(value).toLocaleString();
const indexingStatusLabel = (status) =>
  status === "inprogress" ? "In progress" : status;
const indexingLabel = (doc) => {
  const progress =
    doc.indexing_status === "inprogress" &&
    typeof doc.indexing_progress === "number"
      ? ` ${doc.indexing_progress}%`
      : "";
  return `${indexingStatusLabel(doc.indexing_status)}${progress}`;
};
const documentStatus = (message = "") => {
  byId("document-status").textContent = message;
};
const selectedDocumentIds = new Set();
let displayedDocumentIds = [];

function updateBulkDeleteControl() {
  const deleteSelected = byId("delete-selected");
  const selectAll = byId("select-all-documents");
  const selectedCount = selectedDocumentIds.size;
  deleteSelected.disabled = selectedCount === 0;
  deleteSelected.textContent = selectedCount
    ? `Delete selected (${selectedCount})`
    : "Delete selected";
  selectAll.checked =
    displayedDocumentIds.length > 0 &&
    displayedDocumentIds.every((id) => selectedDocumentIds.has(id));
  selectAll.indeterminate =
    selectedCount > 0 && !selectAll.checked;
}

document
  .querySelectorAll("[data-close]")
  .forEach(
    (button) => (button.onclick = () => byId(button.dataset.close).close()),
  );
byId("open-documents").onclick = async () => {
  openModal("documents-modal");
  documentStatus();
  await docs();
};
byId("open-upload").onclick = () => {
  byId("documents-modal").close();
  openModal("upload-modal");
};
byId("select-all-documents").onchange = (event) => {
  displayedDocumentIds.forEach((id) => {
    if (event.target.checked) selectedDocumentIds.add(id);
    else selectedDocumentIds.delete(id);
  });
  updateBulkDeleteControl();
  document.querySelectorAll(".doc-select").forEach((checkbox) => {
    checkbox.checked = selectedDocumentIds.has(Number(checkbox.value));
  });
};
byId("delete-selected").onclick = () => deleteSelectedDocuments();

async function docs() {
  const list = byId("documents");
  try {
    const response = await fetch("/api/documents?limit=500");
    if (!response.ok) throw new Error("Could not load documents.");
    const data = await response.json();
    const documents = data.documents || [];
    displayedDocumentIds = documents.map((doc) => doc.id);
    selectedDocumentIds.forEach((id) => {
      if (!displayedDocumentIds.includes(id)) selectedDocumentIds.delete(id);
    });
    byId("count").textContent =
      `${documents.length} document${documents.length === 1 ? "" : "s"} available`;
    list.replaceChildren();
    if (!documents.length) {
      const empty = document.createElement("p");
      empty.className = "subtle";
      empty.textContent = "No documents have been uploaded yet.";
      list.append(empty);
      updateBulkDeleteControl();
      return;
    }

    documents.forEach((doc) => {
      const row = document.createElement("div");
      row.className = "doc-row";
      const select = document.createElement("input");
      select.className = "doc-select";
      select.type = "checkbox";
      select.value = doc.id;
      select.checked = selectedDocumentIds.has(doc.id);
      select.setAttribute("aria-label", `Select ${doc.filename}`);
      select.addEventListener("change", () => {
        if (select.checked) selectedDocumentIds.add(doc.id);
        else selectedDocumentIds.delete(doc.id);
        updateBulkDeleteControl();
      });
      const details = document.createElement("span");
      details.className = "doc-details";
      const filename = document.createElement("b");
      filename.className = "doc-filename";
      filename.textContent = doc.filename;
      filename.title = doc.filename;
      const metadata = document.createElement("small");
      metadata.textContent = `${documentTypeLabel(doc.doc_type)} · ${formatBytes(doc.byte_size)} · ${formatDate(doc.created_at)}`;
      details.append(filename, metadata);
      if (doc.indexing_status === "failed" && doc.indexing_error) {
        const error = document.createElement("small");
        error.className = "indexing-error";
        error.textContent = doc.indexing_error;
        error.title = doc.indexing_error;
        details.append(error);
      }
      const actions = document.createElement("span");
      actions.className = "doc-actions";
      const indexingStatus = document.createElement("span");
      indexingStatus.className = `indexing-status is-${doc.indexing_status}`;
      indexingStatus.textContent = indexingLabel(doc);
      if (doc.indexing_error) indexingStatus.title = doc.indexing_error;
      const deleteButton = document.createElement("button");
      deleteButton.className = "delete-document";
      deleteButton.type = "button";
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", () =>
        deleteDocument(doc.id, doc.filename, deleteButton),
      );
      actions.append(indexingStatus, deleteButton);
      row.append(select, details, actions);
      list.append(row);
    });
    updateBulkDeleteControl();
  } catch (error) {
    byId("count").textContent = "Documents unavailable";
    list.innerHTML = "";
    const message = document.createElement("p");
    message.className = "subtle";
    message.textContent = error.message;
    list.append(message);
  }
}
async function deleteDocument(id, filename, button) {
  if (!window.confirm(`Delete ${filename}?`)) return;

  button.disabled = true;
  documentStatus("Deleting document…");
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Could not delete the document.");
    selectedDocumentIds.delete(id);
    documentStatus("Document deleted.");
    await docs();
  } catch (error) {
    documentStatus(error.message);
    button.disabled = false;
  }
}
async function deleteSelectedDocuments() {
  const documentIds = [...selectedDocumentIds];
  if (!documentIds.length) return;
  if (!window.confirm(`Delete ${documentIds.length} selected documents?`)) return;

  const button = byId("delete-selected");
  button.disabled = true;
  documentStatus("Deleting selected documents…");
  try {
    const response = await fetch("/api/documents", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: documentIds }),
    });
    if (!response.ok) throw new Error("Could not delete the selected documents.");
    selectedDocumentIds.clear();
    documentStatus("Selected documents deleted.");
    await docs();
  } catch (error) {
    documentStatus(error.message);
    updateBulkDeleteControl();
  }
}
byId("upload-form").onsubmit = async (event) => {
  event.preventDefault();
  const status = byId("upload-status");
  const submit = byId("upload");
  const files = [...byId("content").files];
  if (!files.length) {
    status.textContent = "Select one or more documents to upload.";
    return;
  }
  status.textContent = `Processing ${files.length} document${files.length === 1 ? "" : "s"}…`;
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("doc_type", byId("type").value);
  submit.disabled = true;
  try {
    const response = await fetch("/api/documents", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) {
      status.textContent = data.detail || "Upload failed.";
      return;
    }
    const uploadedMessage = `${data.documents.length} document${data.documents.length === 1 ? "" : "s"} uploaded.`;
    status.textContent = uploadedMessage;
    event.target.reset();
    byId("upload-modal").close();
    openModal("documents-modal");
    documentStatus(uploadedMessage);
    await docs();
  } catch {
    status.textContent = "Upload failed. Please try again.";
  } finally {
    submit.disabled = false;
  }
};
docs();
