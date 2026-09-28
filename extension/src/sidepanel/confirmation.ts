import type {
  ConfirmationRequestPayload,
  ConfirmationResponsePayload,
} from "@shared/protocol";

type Decision = ConfirmationResponsePayload["decision"];

function required<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element as unknown as T;
}

export class ConfirmationController {
  private readonly dialog = required<HTMLDialogElement>("confirmation-dialog");
  private readonly tool = required<HTMLElement>("confirmation-tool");
  private readonly target = required<HTMLElement>("confirmation-target");
  private readonly reason = required<HTMLElement>("confirmation-reason");
  private readonly arguments = required<HTMLElement>("confirmation-arguments");
  private readonly risks = required<HTMLElement>("confirmation-risks");
  private paused = false;
  private oneShotOnly = false;
  private resolve: ((decision: Decision) => void) | null = null;

  constructor() {
    this.dialog.addEventListener("close", () => {
      const decision = this.dialog.returnValue;
      this.finish(
        decision === "allow_once" || decision === "allow_run"
          ? (this.paused ? "cancel" : decision)
          : "cancel",
      );
    });
    this.dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      this.dialog.close("cancel");
    });
  }

  ask(payload: ConfirmationRequestPayload): Promise<Decision> {
    this.cancel();
    this.oneShotOnly = payload.tool === "close_tab" || payload.tool === "acquire_attachment" || payload.tool === "open_attachment_preview" || payload.risk_reasons.includes("delete");
    this.setPaused(this.paused);
    this.tool.textContent = payload.tool;
    this.target.textContent = payload.target || "(no element target)";
    this.reason.textContent = payload.reason;
    this.arguments.textContent = JSON.stringify(payload.arguments, null, 2);
    this.risks.textContent = payload.risk_reasons.join(", ") || "high risk";
    this.dialog.showModal();
    return new Promise((resolve) => {
      this.resolve = resolve;
    });
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    for (const button of this.dialog.querySelectorAll<HTMLButtonElement>("button[value=allow_once],button[value=allow_run]")) {
      button.disabled = paused || (this.oneShotOnly && button.value === "allow_run");
    }
  }

  cancel(): void {
    if (this.dialog.open) this.dialog.close("cancel");
    else this.finish("cancel");
  }

  reset(): void {
    this.cancel();
    for (const element of [this.tool, this.target, this.reason, this.arguments, this.risks]) element.textContent = "";
    this.oneShotOnly = false;
    this.setPaused(false);
  }

  private finish(decision: Decision): void {
    const resolve = this.resolve;
    this.resolve = null;
    resolve?.(decision);
  }
}
