import { useLingui } from '@lingui/react/macro';

interface ViewSelectorProps {
  selectedView: number;
  onViewChange: (view: number) => void;
}

export default function ViewSelector({
  selectedView,
  onViewChange
}: ViewSelectorProps) {
  const { t } = useLingui();
  const views = [
    { id: 0, label: t`Encoding Info`, icon: 'fa-binary-lock' },
    { id: 1, label: t`Correlation Matrix`, icon: 'fa-hashtag' },
    { id: 2, label: t`Missing Values Heatmap`, icon: 'fa-value-absolute' },
    { id: 3, label: t`Data Distribution`, icon: 'fa-chart-simple' }
  ];

  return (
    <div className='viewButtons'>
      {views.map((view) => (
        <button
          key={view.id}
          onClick={() => onViewChange(view.id)}
          style={
            selectedView === view.id
              ? {
                  backgroundColor: 'var(--primary-color)',
                  color: 'var(--card-background)'
                }
              : {}
          }
        >
          <i className={`fa-solid ${view.icon}`}></i> {view.label}
        </button>
      ))}
    </div>
  );
}
